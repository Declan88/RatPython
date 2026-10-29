"""
The "Loading..." screen: centred text with an animated rat in the bottom-left corner.

Loading is still fully synchronous - this project's OpenGL context can only ever be touched from the
one thread that created it, so nothing can animate by itself while a scene builds. Instead the screen
stays alive for the whole load and is redrawn from the loading code: Scene's _pump_load() (called
before each object loads and between lightmap bakes) calls tick(), which redraws at most every
FRAME_INTERVAL seconds. The animation is chosen from the clock, not a frame counter, so it keeps its
speed however unevenly the redraws arrive. A single long step (a big lightmap bake, say) still holds
the picture still until the next pump.

    show_loading_screen(window, "Loading...")   # starts it (or changes its text) and draws a frame
    tick()                                      # redraws if due; does nothing when no screen is up
    hide_loading_screen()                       # when the load is over - releases everything

The rat is Assets/Textures/Icons/loadingsheet.png: a SHEET_COLS x SHEET_ROWS grid of equal frames read
left to right, top to bottom, SHEET_FRAMES of them (the last row is not full).

No text-rendering pipeline exists elsewhere in this project (every other visual is a 3D scene) - the
text is built with pygame.font, rendered to a Surface and uploaded as a texture.
"""

import time

import moderngl
import numpy as np
import pygame

SHEET_PATH = "Assets/Textures/Icons/loadingsheet.png"
SHEET_COLS, SHEET_ROWS, SHEET_FRAMES = 4, 5, 18
FPS = 18.0                    # animation speed: the 18 frames play once a second
SPRITE_SCALE = 0.18           # the rat's height as a fraction of the window's HEIGHT (its width follows its shape)
SPRITE_MARGIN = 0.03          # the gap from the rat to the left and bottom edges, as a fraction of the window's height
FRAME_INTERVAL = 1.0 / 30.0   # tick() redraws at most this often

_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv);
}
"""

_active = None


def _texture(ctx, surface):
    # flipped=True - GL textures are bottom-up, a pygame Surface's own byte order is top-down.
    tex = ctx.texture(surface.get_size(), 4, pygame.image.tostring(surface, "RGBA", True))
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    return tex


def _quad(cx, cy, half_w, half_h, u0=0.0, v0=0.0, u1=1.0, v1=1.0):
    """Two triangles centred on (cx, cy) in NDC, as (x, y, u, v) rows."""
    x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h
    return np.array([
        x0, y0, u0, v0,   x1, y0, u1, v0,   x1, y1, u1, v1,
        x0, y0, u0, v0,   x1, y1, u1, v1,   x0, y1, u0, v1,
    ], dtype="f4")


def _content_bounds(surface):
    """The box (left, top, right, bottom) in pixels INSIDE a frame cell that the rat occupies in at least
    one frame - the frames have transparent padding, and anchoring the whole cell to the corner would
    leave a gap. One box for every frame, so the rat doesn't shift as the animation plays."""
    width, height = surface.get_size()
    alpha = np.frombuffer(pygame.image.tostring(surface, "RGBA"), dtype=np.uint8).reshape(height, width, 4)[..., 3]
    cell_w, cell_h = width // SHEET_COLS, height // SHEET_ROWS
    left, top, right, bottom = cell_w, cell_h, 0, 0
    for frame in range(SHEET_FRAMES):
        col, row = frame % SHEET_COLS, frame // SHEET_COLS
        ys, xs = np.nonzero(alpha[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] > 8)
        if len(xs):
            left, right = min(left, xs.min()), max(right, xs.max() + 1)
            top, bottom = min(top, ys.min()), max(bottom, ys.max() + 1)
    if right <= left or bottom <= top:
        return 0, 0, cell_w, cell_h
    return int(left), int(top), int(right), int(bottom)


class _Screen:
    def __init__(self, window, text):
        self.window = window
        self.ctx = window.ctx
        self.last_draw = 0.0
        self.text = None
        self._text_tex = self._text_vbo = self._text_vao = None
        self.set_text(text)

        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.sheet = None
        try:
            image = pygame.image.load(SHEET_PATH).convert_alpha()
            self.sheet = _texture(self.ctx, image)
            cell_w, cell_h = image.get_width() // SHEET_COLS, image.get_height() // SHEET_ROWS
            left, top, right, bottom = _content_bounds(image)
            # The rat's box as fractions of a cell, and its shape (width / height).
            self.crop = (left / cell_w, top / cell_h, right / cell_w, bottom / cell_h)
            self.crop_aspect = (right - left) / (bottom - top)
        except (pygame.error, FileNotFoundError):
            pass                  # no rat - the text alone still shows
        # The rat's quad is rewritten every draw (its UVs pick the frame), the text's only on a change.
        self.sprite_vbo = self.ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self.sprite_vao = self.ctx.vertex_array(self.prog, [(self.sprite_vbo, "2f 2f", "in_pos", "in_uv")])

    def set_text(self, text):
        if text == self.text:
            return
        self.text = text
        for res in (self._text_vao, self._text_vbo, self._text_tex):
            if res is not None:
                res.release()
        if not pygame.font.get_init():
            pygame.font.init()
        surface = pygame.font.SysFont(None, 48).render(text, True, (230, 230, 230)).convert_alpha()
        self._text_tex = _texture(self.ctx, surface)
        self._text_size = surface.get_size()
        self._text_vbo = self.ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self._text_vao = None     # built with the program in _ensure_text_vao

    def _ensure_text_vao(self):
        if self._text_vao is None:
            self._text_vao = self.ctx.vertex_array(self.prog, [(self._text_vbo, "2f 2f", "in_pos", "in_uv")])

    def draw(self):
        ctx, window = self.ctx, self.window
        # The framebuffer's real size, NOT window.width/height: those only change when a resize event is
        # handled, and a load blocks the event loop - after a resize or maximise they are stale, which put
        # everything in the wrong place.
        width, height = ctx.screen.size
        self._ensure_text_vao()
        # Sizes in NDC from pixels, so neither the text nor the rat is stretched by the window's shape.
        text_w, text_h = self._text_size
        self._text_vbo.write(_quad(0.0, 0.0, text_w / width, text_h / height).tobytes())

        ctx.screen.use()
        ctx.viewport = (0, 0, width, height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.clear(0.05, 0.05, 0.05, 1.0)
        self.prog["u_tex"].value = 0
        self._text_tex.use(location=0)
        self._text_vao.render(moderngl.TRIANGLES)

        if self.sheet is not None:
            frame = int(time.perf_counter() * FPS) % SHEET_FRAMES
            col, row = frame % SHEET_COLS, frame // SHEET_COLS
            # Only the rat's own box of the cell is drawn. Row 0 is the TOP of the image, which the upload
            # flipped to the top of the texture (v = 1).
            c_left, c_top, c_right, c_bottom = self.crop
            u0, u1 = (col + c_left) / SHEET_COLS, (col + c_right) / SHEET_COLS
            v1, v0 = 1.0 - (row + c_top) / SHEET_ROWS, 1.0 - (row + c_bottom) / SHEET_ROWS
            # Anchored so the rat's own lower-left corner sits SPRITE_MARGIN in from the window's left and
            # bottom edges; SPRITE_SCALE is the rat's height. Sized from the window height, so it scales.
            sprite_h = SPRITE_SCALE * height
            sprite_w = sprite_h * self.crop_aspect
            margin = SPRITE_MARGIN * height
            cx = -1.0 + 2.0 * (margin + sprite_w / 2) / width
            cy = -1.0 + 2.0 * (margin + sprite_h / 2) / height
            # _quad takes HALF sizes in NDC, and NDC spans 2 units over the window: half = pixels / size.
            self.sprite_vbo.write(_quad(cx, cy, sprite_w / width, sprite_h / height,
                                        u0, v0, u1, v1).tobytes())
            self.sheet.use(location=0)
            self.sprite_vao.render(moderngl.TRIANGLES)

        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        window.flip()
        # Doesn't make the load itself responsive, but stops Windows flagging the window "Not Responding".
        pygame.event.pump()
        self.last_draw = time.perf_counter()

    def release(self):
        for res in (self.sprite_vao, self.sprite_vbo, self.sheet, self._text_vao, self._text_vbo,
                    self._text_tex, self.prog):
            if res is not None:
                res.release()


def show_loading_screen(window, text="Loading..."):
    """Shows the loading screen with `text` (starting it, or just changing the text of the one already
    up) and draws a frame. Call hide_loading_screen() when the load is over."""
    global _active
    if _active is not None and _active.window is not window:
        hide_loading_screen()
    if _active is None:
        _active = _Screen(window, text)
    else:
        _active.set_text(text)
    _active.draw()


def tick():
    """Redraws the loading screen if one is up and a frame is due; otherwise does nothing. Returns True
    if it drew a frame - which CHANGES OpenGL state (framebuffer, viewport, blending, depth test, culling),
    so code in the middle of a GL pass that relies on its own state must set it again after a True."""
    if _active is not None and time.perf_counter() - _active.last_draw >= FRAME_INTERVAL:
        _active.draw()
        return True
    return False


def hide_loading_screen():
    global _active
    if _active is not None:
        _active.release()
        _active = None
