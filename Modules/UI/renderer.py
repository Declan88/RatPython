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

# The ordinary, undistorted 0..1 UV mapping for a quad's (top-left, top-right,
# bottom-right, bottom-left) corners - textures are uploaded flipped (see
# texture_from_surface), so the quad's top edge samples v=1.
_STANDARD_UV = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))

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
        # (texture, corners, uvs, (r, g, b, a), clip) - corners/uvs are each ((x0,y0),
        # (x1,y0), (x1,y1), (x0,y1))-shaped 4-tuples (pixel space for corners, 0..1
        # texture space for uvs) in this same (top-left, top-right, bottom-right,
        # bottom-left) winding - kept as explicit points rather than a plain
        # (x0,y0,x1,y1) rect so texture_logical_oriented (the damage indicator's
        # rotating arrow) and texture_logical_cover (the scope overlay's own aspect-
        # preserving fit - see its own docstring) can both hand back a non-trivial quad
        # through this exact same draw()/batching path with no special-casing, instead
        # of needing a separate kind of quad per caller. uvs defaults to _STANDARD_UV
        # (0..1 the ordinary way) for every axis-aligned/rotated-but-undistorted call;
        # only texture_logical_cover ever hands back something else.
        self.quads = []
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

    @staticmethod
    def _axis_aligned_corners(x0, y0, x1, y1):
        return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

    def rect(self, logical_rect, color):
        x, y, w, h = logical_rect
        s = self.scale
        x0, y0 = round(x * s), round(y * s)
        x1, y1 = round((x + w) * s), round((y + h) * s)
        self.quads.append((self.white, self._axis_aligned_corners(x0, y0, x1, y1), _STANDARD_UV, color, self.clip))

    def texture(self, tex, px, py, pw, ph, color):
        x0, y0 = round(px), round(py)
        self.quads.append(
            (tex, self._axis_aligned_corners(x0, y0, x0 + pw, y0 + ph), _STANDARD_UV, color, self.clip))

    def texture_logical(self, tex, logical_rect, color):
        x, y, w, h = logical_rect
        s = self.scale
        x0, y0 = round(x * s), round(y * s)
        x1, y1 = round((x + w) * s), round((y + h) * s)
        self.quads.append((tex, self._axis_aligned_corners(x0, y0, x1, y1), _STANDARD_UV, color, self.clip))

    def texture_logical_cover(self, tex, logical_rect, image_size, color):
        """Like texture_logical, but preserves the TEXTURE's own aspect ratio instead of
        stretching it to logical_rect's shape: fits image_size (the texture's own native
        (w, h), any units - only the ratio matters) undistorted against whichever axis of
        logical_rect actually constrains it, then EXTENDS to fully cover the other axis
        by letting that axis's own UV range run outside [0, 1] - relying on the texture's
        own CLAMP_TO_EDGE wrap mode (set this on `tex` - see texture_from_surface's own
        repeat_x/y, not the default REPEAT) to repeat its outermost pixel row/column
        across the leftover space, rather than wrapping back around to the image's
        opposite edge. Built for a full-screen overlay authored at one fixed aspect ratio
        (the scope reticle's own 1920x1080) that has to cover an arbitrary window aspect
        without visibly warping its own centred artwork - a solid-coloured border (this
        kind of overlay's own vignette) extending this way reads as seamless, unlike
        texture_logical's own non-uniform stretch or a plain crop-to-cover fit.

        image_size: (w, h) - the texture's own pixel dimensions (or any value in the
        same ratio; only width/height matters, not the absolute scale)."""
        x, y, w, h = logical_rect
        s = self.scale
        x0, y0 = round(x * s), round(y * s)
        x1, y1 = round((x + w) * s), round((y + h) * s)
        screen_w, screen_h = x1 - x0, y1 - y0
        img_w, img_h = image_size
        if img_w <= 0 or img_h <= 0 or screen_w <= 0 or screen_h <= 0:
            return
        # The scale that fits image_size inside (screen_w, screen_h) undistorted and
        # without cropping - exactly one axis comes out equal to its own screen size
        # (the constraining one); the other comes out smaller, leaving a gap.
        fit_scale = min(screen_w / img_w, screen_h / img_h)
        fitted_w, fitted_h = img_w * fit_scale, img_h * fit_scale
        # Per axis: half the screen size expressed in "fitted images" (0.5 exactly on
        # the constraining axis, since screen_size == fitted_size there -> UV stays
        # [0, 1] unchanged; > 0.5 on the gap axis, extending UV beyond [0, 1] by exactly
        # how much gap there is, so the image is still centred).
        half_u = (screen_w / fitted_w) / 2.0
        half_v = (screen_h / fitted_h) / 2.0
        u0, u1 = 0.5 - half_u, 0.5 + half_u
        v0, v1 = 0.5 - half_v, 0.5 + half_v
        uvs = ((u0, v1), (u1, v1), (u1, v0), (u0, v0))
        self.quads.append((tex, self._axis_aligned_corners(x0, y0, x1, y1), uvs, color, self.clip))

    def texture_logical_oriented(self, tex, center_logical, forward, logical_size, color):
        """Like texture_logical, but the quad is built from an explicit 2D unit `forward`
        vector (+x right, +y down, same as everything else in this screen-space UI) instead
        of always coming out axis-aligned - for a widget that needs to rotate freely to
        point in an arbitrary on-screen direction (the damage indicator's arrow - see
        damage_indicator.py), which no other caller here needs (see DrawList's own
        docstring on why every other quad stays a plain rect).

        center_logical: the quad's own centre, in logical units (not a corner, unlike
        every other *_logical call here - a rotating quad has no stable "top-left").
        forward: (fx, fy) - the direction the TEXTURE's own top edge (where its content
        should read "up") points on screen; the caller normalizes it, not this function
        (so (0, 0) - all the directional calcs below would divide-by-zero-adjacent
        degenerate into a zero-size quad - never gets handed to a live draw this way in
        practice, since damage_indicator.py always has a real direction to show by the
        time it draws). logical_size: (w, h), same convention as every other *_logical
        call's own size."""
        cx, cy = center_logical
        w, h = logical_size
        s = self.scale
        fx, fy = forward
        rx, ry = -fy, fx   # `forward` rotated 90 degrees clockwise on screen - the quad's own local +X (right)
        ccx, ccy = cx * s, cy * s
        hw, hh = (w * s) / 2.0, (h * s) / 2.0
        top = (ccx + fx * hh, ccy + fy * hh)
        bottom = (ccx - fx * hh, ccy - fy * hh)
        corners = (
            (top[0] - rx * hw, top[1] - ry * hw),
            (top[0] + rx * hw, top[1] + ry * hw),
            (bottom[0] + rx * hw, bottom[1] + ry * hw),
            (bottom[0] - rx * hw, bottom[1] - ry * hw),
        )
        self.quads.append((tex, corners, _STANDARD_UV, color, self.clip))


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
        for _, corners, uvs, c, _clip in quads:
            # corners/uvs are already in this exact (top-left, top-right, bottom-right,
            # bottom-left) winding for every quad, axis-aligned or not - see DrawList's
            # own docstring.
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners
            (u0, v0), (u1, v1), (u2, v2), (u3, v3) = uvs
            verts.extend((
                x0, y0, u0, v0, *c,  x1, y1, u1, v1, *c,  x2, y2, u2, v2, *c,
                x0, y0, u0, v0, *c,  x2, y2, u2, v2, *c,  x3, y3, u3, v3, *c,
            ))
        self.vbo.orphan()     # a fresh buffer: writing the one the GPU may still be reading would stall
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
                    or quads[i][4] != quads[start][4]):
                clip = quads[start][4]
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
