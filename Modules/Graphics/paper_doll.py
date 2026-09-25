"""
A tiny "paper doll" of the player's head in a screen corner (Minecraft's
inventory-style head), drawn as cheaply as possible:

  * It re-draws the player's ALREADY-animated skinned model - no second
    skeleton, no extra animation update, no extra buffers: the doll builds
    its own VAO over the model's existing vertex buffers, and skins from
    the same per-frame bone uniform buffer the main pass uploaded.
  * Its own minimal program (skinning + texture * one fixed light) instead of
    the full PBR path - no shadows, probes, point lights, SSR, lightmaps.
  * One draw call into a small scissored viewport, so the GPU fragment cost
    is a few thousand pixels.
"""

import glm
import moderngl

from Modules.Graphics.skeletal_shader import MAX_BONES, _bind_bone_block, bind_bone_matrices

_VERTEX = f"""
#version 330
#define MAX_BONES {MAX_BONES}
uniform mat4 u_mvp;
uniform mat3 u_normal_rot;
layout(std140) uniform BoneBlock {{ mat4 u_bone_matrices[MAX_BONES]; }};
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in ivec4 in_joints;
in vec4 in_weights;
out vec3 v_normal;
out vec2 v_uv;
void main() {{
    mat4 skin =
        in_weights.x * u_bone_matrices[in_joints.x] +
        in_weights.y * u_bone_matrices[in_joints.y] +
        in_weights.z * u_bone_matrices[in_joints.z] +
        in_weights.w * u_bone_matrices[in_joints.w];
    v_normal = u_normal_rot * (mat3(skin) * in_normal);
    v_uv = in_uv;
    gl_Position = u_mvp * (skin * vec4(in_position, 1.0));
}}
"""

_FRAGMENT = """
#version 330
uniform sampler2D u_texture;
uniform int u_has_texture;
uniform vec4 u_box;         // box origin xy + size zw, window pixels
uniform vec3 u_circle;      // box uv centre xy, radius (fraction of box height)
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 base = u_has_texture == 1 ? texture(u_texture, v_uv) : vec4(0.7, 0.7, 0.7, 1.0);
    if (base.a < 0.5) discard;
    // Below the badge's centre line only the part inside the circle shows, so
    // the body is cropped by the circle while the head above it pokes out.
    vec2 uv = (gl_FragCoord.xy - u_box.xy) / u_box.zw;
    if (uv.y < u_circle.y && length(uv - u_circle.xy) > u_circle.z) discard;
    float light = 0.55 + 0.45 * max(dot(normalize(v_normal), normalize(vec3(0.3, 0.5, 1.0))), 0.0);
    f_color = vec4(base.rgb * light, 1.0);
}
"""

_CIRCLE_VERTEX = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);   // fullscreen triangle
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_CIRCLE_FRAGMENT = """
#version 330
uniform vec2 u_center;      // box uv (0..1)
uniform float u_radius;     // box-height fraction
uniform float u_aspect;     // box w / h (1.0 here, kept for clarity)
uniform vec4 u_fill;
uniform vec4 u_rim;
uniform float u_rim_width;  // box-height fraction
in vec2 v_uv;
out vec4 f_color;
void main() {
    float d = length((v_uv - u_center) * vec2(u_aspect, 1.0));
    float aa = 0.006;
    float inside = 1.0 - smoothstep(u_radius - aa, u_radius + aa, d);
    float rim = smoothstep(u_radius - u_rim_width - aa, u_radius - u_rim_width + aa, d);
    vec4 c = mix(u_fill, u_rim, rim);
    f_color = vec4(c.rgb, c.a * inside);
}
"""

_HEAD_JOINT = "ValveBiped.Bip01_Head1"

REFERENCE_HEIGHT = 1080.0  # logical layout height, same convention as the UI system


class PaperDoll:
    """The head is drawn over a round badge and is deliberately taller than
    it, so it pokes out of the top of the circle. All sizes are logical pixels
    (scaled by window height / 1080 like the UI), measured from the
    bottom-left of the window."""

    def __init__(self, ctx, obj, center=(78.0, 62.0), radius=38.0, box_size=140.0,
                 circle_center_frac=0.36, head_center_frac=0.56, half_height=0.5,
                 yaw_degrees=20.0):
        self.ctx = ctx
        self.obj = obj
        self.center = center
        self.radius = radius
        self.box_size = box_size
        self.circle_cy = circle_center_frac   # circle centre, fraction up the box
        self.head_cy = head_center_frac       # head centre, fraction up the box
        self.half_height = half_height
        self.yaw = glm.radians(yaw_degrees)
        self.fill = (0.08, 0.09, 0.12, 0.85)
        self.rim = (0.75, 0.78, 0.85, 1.0)

        self.program = _bind_bone_block(ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT))
        vbos = obj["_render_vbos"]
        content = [
            (vbos["in_position"], "3f", "in_position"),
            (vbos["in_normal"], "3f", "in_normal"),
            (vbos["in_uv"], "2f", "in_uv"),
            (vbos["in_joints"], "4i", "in_joints"),
            (vbos["in_weights"], "4f", "in_weights"),
        ]
        self.vao = ctx.vertex_array(self.program, content, obj["_render_ibo"])
        self._hat_vaos = {}   # hat name -> VAO over that hat's existing buffers
        self.circle_program = ctx.program(vertex_shader=_CIRCLE_VERTEX, fragment_shader=_CIRCLE_FRAGMENT)
        self.circle_vao = ctx.vertex_array(self.circle_program, [])

        # Head's bind-pose position in mesh space; skinning it with the
        # current bone matrix follows the head as the pose moves/crouches.
        joints = obj["skeleton"].joints
        self._head_index = next((i for i, j in enumerate(joints) if j.name == _HEAD_JOINT), None)
        if self._head_index is not None:
            bind = glm.inverse(joints[self._head_index].inverse_bind_matrix)
            self._head_bind = glm.vec4(bind[3].x, bind[3].y, bind[3].z, 1.0)

    def _head_position(self):
        if self._head_index is None:
            return glm.vec3(0.0, 1.3, 0.0)
        bones = self.obj["bone_matrices"]
        p = bones[self._head_index] * self._head_bind
        return glm.vec3(p) * self.obj["scale"]

    def render(self, window_size):
        if self.obj is None or self.obj.get("bone_ubo") is None:
            return
        w, h = window_size
        k = h / REFERENCE_HEIGHT
        size = int(self.box_size * k)
        x = int(self.center[0] * k - size * 0.5)
        y = int(self.center[1] * k - size * self.circle_cy)
        box = (x, y, size, size)

        ctx = self.ctx
        ctx.scissor = box
        ctx.viewport = box

        # Badge: a round backdrop, alpha-blended so the window shows through.
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        cp = self.circle_program
        cp["u_center"].value = (0.5, self.circle_cy)
        cp["u_radius"].value = self.radius / self.box_size
        cp["u_aspect"].value = 1.0
        cp["u_fill"].value = self.fill
        cp["u_rim"].value = self.rim
        cp["u_rim_width"].value = 3.0 / self.box_size
        self.circle_vao.render(moderngl.TRIANGLES, vertices=3)

        # Head: depth-only clear inside the box (colour untouched) so it isn't
        # hidden by the scene behind it, then drawn over the badge - it is not
        # clipped to the circle, which is what makes it pop out.
        # (Masks live on the Framebuffer and are only applied to GL state by
        # use()/clear() - so re-use() after restoring, or the head's colour
        # writes stay masked off. use() also resets viewport/scissor, hence
        # setting those again afterwards.)
        screen = ctx.screen
        screen.use()
        screen.color_mask = (False, False, False, False)
        screen.clear(depth=1.0, viewport=box)
        screen.color_mask = (True, True, True, True)
        screen.use()
        ctx.scissor = box
        ctx.viewport = box
        ctx.enable(moderngl.DEPTH_TEST)

        head = self._head_position()
        rot = glm.rotate(glm.mat4(1.0), self.yaw, glm.vec3(0.0, 1.0, 0.0))
        head = glm.vec3(rot * glm.vec4(head, 1.0))
        eye = head + glm.vec3(0.0, 0.0, 2.0)
        view = glm.lookAt(eye, head, glm.vec3(0.0, 1.0, 0.0))
        r = self.half_height
        # Vertical extent shifted so the head centre lands head_cy up the box.
        proj = glm.ortho(-r, r, -2.0 * r * self.head_cy, 2.0 * r * (1.0 - self.head_cy), 0.1, 5.0)
        model = rot * glm.scale(glm.mat4(1.0), self.obj["scale"])

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.CULL_FACE)
        ctx.cull_face = "back"

        self.program["u_mvp"].write((proj * view * model).to_bytes())
        self.program["u_normal_rot"].write(glm.mat3(rot).to_bytes())
        self.program["u_box"].value = (float(x), float(y), float(size), float(size))
        self.program["u_circle"].value = (0.5, self.circle_cy, self.radius / self.box_size)
        tex = self.obj.get("texture")
        has_tex = 1 if tex is not None else 0
        if has_tex:
            tex.use(location=0)
            self.program["u_texture"].value = 0
        self.program["u_has_texture"].value = has_tex
        bind_bone_matrices(self.obj)
        self.vao.render()
        hat = self.obj["hats"].get(self.obj.get("active_hat"))
        if hat is not None:
            name = self.obj["active_hat"]
            vao = self._hat_vaos.get(name)
            if vao is None:
                v = hat["vbos"]
                vao = self._hat_vaos[name] = self.ctx.vertex_array(self.program, [
                    (v["in_position"], "3f", "in_position"), (v["in_normal"], "3f", "in_normal"),
                    (v["in_uv"], "2f", "in_uv"), (v["in_joints"], "4i", "in_joints"),
                    (v["in_weights"], "4f", "in_weights"),
                ], hat["ibo"])
            if hat["texture"] is not None:
                hat["texture"].use(location=0)
            vao.render()

        ctx.scissor = None
        ctx.viewport = (0, 0, w, h)
