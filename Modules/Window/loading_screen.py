"""
Minimal one-off "Loading..." screen - shown for exactly one frame right
before a scene's own (still fully synchronous/blocking) construction
runs, so switching to a not-yet-built scene reads as "loading" instead
of as a frozen/crashed window. Deliberately NOT a real async/responsive
loading screen: this project's OpenGL context can only ever be touched
from the one thread that created it (see app.py's own scene-loading
comments), so there's no way to keep rendering DURING the actual load
without a much larger architecture change (a background thread for the
CPU-side work, GPU uploads still batched on the main thread afterward) -
this is the simple, low-risk version of "show something before a
blocking stretch" instead.

No text-rendering pipeline exists elsewhere in this project (every
other visual is a 3D scene) - built fresh here via pygame.font (which
pygame.init() already initializes) rendered to a Surface, uploaded as a
one-off moderngl texture, and drawn as a single centered textured quad
in raw NDC space (no camera/projection needed for a single fullscreen-
ish 2D element). Everything created here is released again before
returning - this runs rarely (once per not-yet-visited scene, once per
app run), so there's no reason to cache/reuse a program or font across
calls.
"""

import pygame
import moderngl
import numpy as np


def show_loading_screen(window, text="Loading..."):
    """Clears the screen, draws `text` centered, and flips - once. Call
    this immediately before constructing a Scene that hasn't been built
    yet (see app.py's get_or_load_scene), so the window shows a loading
    frame instead of just hanging while that construction runs."""
    ctx = window.ctx

    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont(None, 48)
    surf = font.render(text, True, (230, 230, 230)).convert_alpha()
    text_w, text_h = surf.get_size()
    # flipped=True - GL textures are bottom-up, a pygame Surface's own
    # byte order is top-down; without this the text renders upside down.
    data = pygame.image.tostring(surf, "RGBA", True)

    tex = ctx.texture((text_w, text_h), 4, data)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            uniform sampler2D u_tex;
            in vec2 v_uv;
            out vec4 fragColor;
            void main() {
                fragColor = texture(u_tex, v_uv);
            }
        """,
    )

    # Quad sized in NDC to match the text's own pixel aspect ratio
    # (window.width/height are in pixels too - see WindowManager.__init__),
    # so the text isn't stretched to fit an arbitrary box.
    half_w = text_w / window.width
    half_h = text_h / window.height
    verts = np.array([
        -half_w, -half_h, 0.0, 0.0,
         half_w, -half_h, 1.0, 0.0,
         half_w,  half_h, 1.0, 1.0,
        -half_w, -half_h, 0.0, 0.0,
         half_w,  half_h, 1.0, 1.0,
        -half_w,  half_h, 0.0, 1.0,
    ], dtype="f4")
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "2f 2f", "in_pos", "in_uv")])

    ctx.screen.use()
    ctx.viewport = (0, 0, window.width, window.height)
    ctx.disable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.CULL_FACE)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.clear(0.05, 0.05, 0.05, 1.0)
    tex.use(location=0)
    prog["u_tex"].value = 0
    vao.render(moderngl.TRIANGLES)
    ctx.disable(moderngl.BLEND)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    window.flip()

    # Pumps the OS event queue once right after the frame actually shows
    # up - doesn't make the multi-second load itself responsive (nothing
    # can, on this single GL thread - see this module's own docstring),
    # but stops Windows from immediately flagging the window "Not
    # Responding" the instant the blocking construction starts.
    pygame.event.pump()

    vao.release()
    vbo.release()
    tex.release()
    prog.release()
