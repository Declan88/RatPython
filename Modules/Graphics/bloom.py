"""
Selective bloom: only meshes named "bloom_..." in the map glow (see model_loader.is_bloom_node).

The scene's materials tonemap per surface, so there is no HDR frame to threshold - instead the
glowing objects are drawn on their own into a small offscreen buffer, blurred, and added over the
finished frame:

  1. Half-resolution buffer: depth of the frame's opaque geometry (so a wall hides what is behind
     it), then each bloom object in its emissive colour.
  2. Shrunk to quarter resolution and blurred (separable 9-tap Gaussian, BLUR_PASSES round trips).
  3. Added over the screen with a full-screen triangle.

The whole pass is skipped on a frame where no bloom object is visible. Moving objects (players,
props) aren't in the occluder depth, so one standing in front of a bloom object does not hide its
glow.
"""

import math

import moderngl

BLUR_PASSES = 2
COLOR_BOOST = 3.0      # the emissive colour is multiplied by this before blurring (the blur spreads it thin)

_OBJECT_VERT = """
#version 330
uniform mat4 u_mvp;
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

# The glow colour is what the main shader adds as emissive: the flat factor times the emissive
# texture if there is one, plus (for a "self illum" material) the base texture lit by its own alpha.
_OBJECT_FRAG = """
#version 330
uniform vec3 u_emissive;
uniform float u_scale;
uniform int u_has_emissive_tex;
uniform int u_self_illum;
uniform sampler2D u_base_tex;
uniform sampler2D u_emissive_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 color = u_emissive;
    if (u_has_emissive_tex != 0) color *= texture(u_emissive_tex, v_uv).rgb;
    if (u_self_illum != 0) {
        vec4 base = texture(u_base_tex, v_uv);
        color += base.rgb * base.a;
    }
    fragColor = vec4(color * u_scale, 1.0);
}
"""

# A full-screen triangle from gl_VertexID alone - no vertex buffer.
_QUAD_VERT = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_COPY_FRAG = """
#version 330
uniform sampler2D u_tex;
uniform float u_scale;
in vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = vec4(texture(u_tex, v_uv).rgb * u_scale, 1.0); }
"""

_BLUR_FRAG = """
#version 330
uniform sampler2D u_tex;
uniform vec2 u_step;        // one texel along the blur direction, in uv
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 sum = texture(u_tex, v_uv).rgb * 0.227027;
    sum += (texture(u_tex, v_uv + u_step).rgb + texture(u_tex, v_uv - u_step).rgb) * 0.1945946;
    sum += (texture(u_tex, v_uv + u_step * 2.0).rgb + texture(u_tex, v_uv - u_step * 2.0).rgb) * 0.1216216;
    sum += (texture(u_tex, v_uv + u_step * 3.0).rgb + texture(u_tex, v_uv - u_step * 3.0).rgb) * 0.054054;
    sum += (texture(u_tex, v_uv + u_step * 4.0).rgb + texture(u_tex, v_uv - u_step * 4.0).rgb) * 0.016216;
    fragColor = vec4(sum, 1.0);
}
"""


class _Buffer:
    """A colour texture (16-bit float, so the boosted colour isn't clamped) with an optional depth."""

    def __init__(self, ctx, size, with_depth):
        self.size = size
        self.color = ctx.texture(size, 4, dtype="f2")
        self.color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.color.repeat_x = self.color.repeat_y = False
        self.depth = ctx.depth_renderbuffer(size) if with_depth else None
        self.fbo = ctx.framebuffer(color_attachments=[self.color], depth_attachment=self.depth)

    def release(self):
        for res in (self.fbo, self.color, self.depth):
            if res is not None:
                res.release()


class Bloom:
    def __init__(self, ctx):
        self.ctx = ctx
        self._size = None
        self._full = None          # half-resolution: occluder depth + the bloom objects
        self._ping = None          # quarter-resolution blur buffers
        self._pong = None
        self._object_prog = ctx.program(vertex_shader=_OBJECT_VERT, fragment_shader=_OBJECT_FRAG)
        self._copy_prog = ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_COPY_FRAG)
        self._blur_prog = ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_BLUR_FRAG)
        self._quad = ctx.vertex_array(self._copy_prog, [])
        self._blur_quad = ctx.vertex_array(self._blur_prog, [])
        self._object_vaos = []     # [(object dict, vao)] built on first draw, released in destroy

    def _ensure(self, width, height):
        if self._size == (width, height):
            return
        for buf in (self._full, self._ping, self._pong):
            if buf is not None:
                buf.release()
        half = (max(1, width // 2), max(1, height // 2))
        quarter = (max(1, width // 4), max(1, height // 4))
        self._full = _Buffer(self.ctx, half, True)
        self._ping = _Buffer(self.ctx, quarter, False)
        self._pong = _Buffer(self.ctx, quarter, False)
        self._size = (width, height)

    def _vao_for(self, obj):
        vao = obj.get("_bloom_vao")
        if vao is None:
            # model_loader builds 16-bit indices whenever the mesh has <= 65536 vertices (12 bytes each here).
            index_size = 2 if obj["vbo"].size // 12 <= 65536 else 4
            vao = self.ctx.vertex_array(
                self._object_prog, [(obj["vbo"], "3f", "in_position"), (obj["bloom_uv_vbo"], "2f", "in_uv")],
                obj["bloom_ibo"], index_element_size=index_size)
            obj["_bloom_vao"] = vao
            self._object_vaos.append(obj)
        return vao

    def render(self, screen, objects, draw_occluders, view_proj, model_matrix_of, intensity=1.0):
        """Draws the glow of `objects` over `screen` (a framebuffer). draw_occluders(fbo) must draw
        the frame's opaque depth into `fbo`; model_matrix_of(obj) is the object's model matrix.
        An object's own "bloom_intensity" scales its brightness and "bloom_radius" its spread
        (both default 1.0); objects sharing a radius are blurred together, one round per radius."""
        if not objects:
            return
        width, height = screen.size
        if width < 4 or height < 4:
            return
        self._ensure(width, height)
        saved_viewport = self.ctx.viewport
        by_radius = {}
        for obj in objects:
            by_radius.setdefault(round(float(obj.get("bloom_radius", 1.0)), 2), []).append(obj)

        # The scene's depth is drawn once for every group; each group then clears only the colour.
        ctx = self.ctx
        full = self._full
        full.fbo.use()
        full.fbo.depth_mask = True
        full.fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.disable(moderngl.BLEND)
        draw_occluders(full.fbo)
        full.fbo.color_mask = (True, True, True, True)
        for index, (radius, group) in enumerate(by_radius.items()):
            if index:
                full.fbo.use()
                full.fbo.depth_mask = False      # a masked glClear leaves the scene's depth alone
                full.fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._draw_group(group, view_proj, model_matrix_of)
            self._blur(radius)
            self._add_to_screen(screen, saved_viewport, intensity)
        full.fbo.depth_mask = True

    def _draw_group(self, group, view_proj, model_matrix_of):
        ctx = self.ctx
        full = self._full
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.disable(moderngl.CULL_FACE)
        full.fbo.depth_mask = False      # bloom objects are tested against the scene, not each other
        prog = self._object_prog
        prog["u_base_tex"].value = 0
        prog["u_emissive_tex"].value = 1
        for obj in group:
            emissive = obj.get("emissive") or (0.0, 0.0, 0.0)
            prog["u_emissive"].value = tuple(float(c) for c in emissive[:3])
            prog["u_scale"].value = COLOR_BOOST * float(obj.get("bloom_intensity", 1.0))
            has_emissive_tex = bool(obj.get("has_emissive_texture")) and obj.get("emissive_texture") is not None
            prog["u_has_emissive_tex"].value = 1 if has_emissive_tex else 0
            self_illum = bool(obj.get("self_illum")) and obj.get("texture") is not None
            prog["u_self_illum"].value = 1 if self_illum else 0
            if obj.get("texture") is not None:
                obj["texture"].use(0)
            if has_emissive_tex:
                obj["emissive_texture"].use(1)
            prog["u_mvp"].write((view_proj * model_matrix_of(obj)).to_bytes())
            self._vao_for(obj).render()

    def _blur(self, radius):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        # Shrink to quarter resolution (the linear filter averages 2x2 texels).
        self._ping.fbo.use()
        self._full.color.use(0)
        self._copy_prog["u_tex"].value = 0
        self._copy_prog["u_scale"].value = 1.0
        self._quad.render(moderngl.TRIANGLES, vertices=3)

        # A bigger radius widens the kernel's spacing (up to 2 texels, past which the 9 taps would
        # band) and adds blur rounds for the rest.
        spacing = min(radius, 2.0)
        rounds = BLUR_PASSES * max(1, math.ceil(radius / 2.0))
        qw, qh = self._ping.size
        for _ in range(rounds):
            for source, target, step in ((self._ping, self._pong, (spacing / qw, 0.0)),
                                         (self._pong, self._ping, (0.0, spacing / qh))):
                target.fbo.use()
                source.color.use(0)
                self._blur_prog["u_tex"].value = 0
                self._blur_prog["u_step"].value = step
                self._blur_quad.render(moderngl.TRIANGLES, vertices=3)

    def _add_to_screen(self, screen, viewport, intensity):
        ctx = self.ctx
        screen.use()
        ctx.viewport = viewport
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._ping.color.use(0)
        self._copy_prog["u_scale"].value = float(intensity)
        self._quad.render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "back"

    def destroy(self):
        for obj in self._object_vaos:
            vao = obj.pop("_bloom_vao", None)
            if vao is not None:
                vao.release()
        self._object_vaos.clear()
        for buf in (self._full, self._ping, self._pong):
            if buf is not None:
                buf.release()
        self._full = self._ping = self._pong = None
        for res in (self._quad, self._blur_quad, self._object_prog, self._copy_prog, self._blur_prog):
            res.release()
