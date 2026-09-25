"""
Batched GL drawing for the UI. Every widget reduces to textured quads in
pixel space (a solid rect is just a quad sampling a 1x1 white texture, so
there is one shader and one vertex format for everything). UIManager
collects a frame's quads in draw order into a DrawList, and draw() uploads
them in one buffer write and issues one draw call per run of consecutive
quads sharing a texture - all solid panels/buttons in a row are a single
draw call; each Label is its own texture and so its own call.
"""

import moderngl
import numpy as np
import pygame

_VERTEX = """
#version 330
uniform vec2 u_screen;
in vec2 in_pos;
in vec2 in_uv;
in vec4 in_color;
out vec2 v_uv;
out vec4 v_color;
void main() {
    v_uv = in_uv;
    v_color = in_color;
    gl_Position = vec4(in_pos.x / u_screen.x * 2.0 - 1.0, 1.0 - in_pos.y / u_screen.y * 2.0, 0.0, 1.0);
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv) * v_color;
}
"""

_FLOATS_PER_VERTEX = 8


class DrawList:
    """A frame's quads, in draw order. Logical->pixel conversion (scale +
    snapping to whole pixels, which keeps edges and text crisp) happens
    here so widgets only ever deal in logical units."""

    def __init__(self, scale, white_tex):
        self.scale = scale
        self.white = white_tex
        self.quads = []  # (texture, x0, y0, x1, y1, (r, g, b, a), clip)
        self.clip = None  # current clip as pixel (x0, y0, x1, y1), or None
        self._clip_stack = []
        self.overlays = []

    def defer(self, fn):
        """Run fn(draw_list) after everything else is drawn, unclipped -
        for popups that must appear above later siblings."""
        self.overlays.append(fn)

    def push_clip(self, logical_rect):
        """Everything drawn until the matching pop_clip is cut off outside
        this rect (nested clips intersect)."""
        x, y, w, h = logical_rect
        s = self.scale
        x0, y0 = round(x * s), round(y * s)
        x1, y1 = round((x + w) * s), round((y + h) * s)
        if self.clip is not None:
            x0, y0 = max(x0, self.clip[0]), max(y0, self.clip[1])
            x1, y1 = min(x1, self.clip[2]), min(y1, self.clip[3])
        self._clip_stack.append(self.clip)
        self.clip = (x0, y0, max(x0, x1), max(y0, y1))

    def pop_clip(self):
        self.clip = self._clip_stack.pop()

    def rect(self, logical_rect, color):
        x, y, w, h = logical_rect
        s = self.scale
        x0, y0 = round(x * s), round(y * s)
        x1, y1 = round((x + w) * s), round((y + h) * s)
        self.quads.append((self.white, x0, y0, x1, y1, color, self.clip))

    def texture(self, tex, px, py, pw, ph, color):
        x0, y0 = round(px), round(py)
        self.quads.append((tex, x0, y0, x0 + pw, y0 + ph, color, self.clip))

    def texture_logical(self, tex, logical_rect, color):
        x, y, w, h = logical_rect
        s = self.scale
        self.quads.append((tex, round(x * s), round(y * s),
                           round((x + w) * s), round((y + h) * s), color, self.clip))


class UIRenderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.white = ctx.texture((1, 1), 4, b"\xff\xff\xff\xff")
        self._capacity = 0
        self.vbo = None
        self.vao = None
        self._ensure_capacity(256)

    def _ensure_capacity(self, quads):
        if quads <= self._capacity:
            return
        if self.vao is not None:
            self.vao.release()
            self.vbo.release()
        self._capacity = max(quads, self._capacity * 2)
        self.vbo = self.ctx.buffer(reserve=self._capacity * 6 * _FLOATS_PER_VERTEX * 4, dynamic=True)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f 4f", "in_pos", "in_uv", "in_color")]
        )

    def texture_from_surface(self, surf):
        # flipped=True: a pygame Surface is top-down, GL textures bottom-up.
        tex = self.ctx.texture(surf.get_size(), 4, pygame.image.tostring(surf, "RGBA", True))
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def new_draw_list(self, scale):
        return DrawList(scale, self.white)

    def draw(self, draw_list, width, height):
        quads = draw_list.quads
        if not quads:
            return
        self._ensure_capacity(len(quads))

        verts = []
        for _, x0, y0, x1, y1, c, _clip in quads:
            # y is down in UI space, so the quad's top edge (y0) samples the
            # top of the texture (v=1 - textures were uploaded flipped).
            verts.extend((
                x0, y0, 0.0, 1.0, *c,  x1, y0, 1.0, 1.0, *c,  x1, y1, 1.0, 0.0, *c,
                x0, y0, 0.0, 1.0, *c,  x1, y1, 1.0, 0.0, *c,  x0, y1, 0.0, 0.0, *c,
            ))
        self.vbo.write(np.asarray(verts, dtype="f4").tobytes())

        ctx = self.ctx
        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog["u_screen"].value = (float(width), float(height))
        self.prog["u_tex"].value = 0

        # A run ends when the texture OR the clip rect changes - so a
        # ScrollBox's contents cost one extra draw call per distinct texture
        # inside it, and everything outside any clip still batches as before.
        start = 0
        for i in range(1, len(quads) + 1):
            if (i == len(quads) or quads[i][0] is not quads[start][0]
                    or quads[i][6] != quads[start][6]):
                clip = quads[start][6]
                if clip is None:
                    ctx.scissor = None
                else:
                    # GL scissor is bottom-left origin; UI space is top-left.
                    ctx.scissor = (clip[0], height - clip[3], clip[2] - clip[0], clip[3] - clip[1])
                quads[start][0].use(location=0)
                self.vao.render(moderngl.TRIANGLES, vertices=(i - start) * 6, first=start * 6)
                start = i

        ctx.scissor = None
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def destroy(self):
        self.vao.release()
        self.vbo.release()
        self.white.release()
        self.prog.release()
