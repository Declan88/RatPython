"""
An offscreen render target for a scene that needs its own depth: screen-space
reflections read the opaque scene's colour and depth, and the default framebuffer's
depth can't be sampled (or blitted on every GPU - Intel rejects a depth blit between
mismatched formats), so such a scene draws into THIS instead and it is copied to the
screen at the end of the frame (Scene.present).

What it holds, and why:
  * `fbo` - what the scene renders into: a colour and a depth TEXTURE, both multisampled
    with the same sample count as the window (so anti-aliasing is unchanged).
  * `resolved_fbo` / `color_tex` / `depth_tex` - single-sample copies of the same colour and
    depth. Filled by resolve(), only on frames a reflective surface is drawn; the
    reflection shader samples these while the scene keeps drawing into `fbo`.
  * `present_fbo` - the same multisampled colour with NO depth, used to copy to the screen:
    a colour-only blit never involves the mismatched-depth-format case.

The depth texture on both sides is 32-bit float (moderngl's only depth texture format), so
the resolve is a blit between identical formats.
"""

import moderngl


class SceneTarget:
    def __init__(self, ctx, size, samples=0):
        self.ctx = ctx
        self.size = tuple(size)
        samples = max(0, int(samples))
        if samples > 1:
            samples = min(samples, int(ctx.info.get("GL_MAX_SAMPLES", samples)))
        self.samples = samples if samples > 1 else 0

        self.color_ms = ctx.texture(self.size, 4, samples=self.samples)
        self.depth_ms = ctx.depth_texture(self.size, samples=self.samples)
        self.fbo = ctx.framebuffer(color_attachments=[self.color_ms], depth_attachment=self.depth_ms)
        self.present_fbo = ctx.framebuffer(color_attachments=[self.color_ms])

        self.color_tex = ctx.texture(self.size, 4)
        self.color_tex.repeat_x = self.color_tex.repeat_y = False
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_tex = ctx.depth_texture(self.size)
        self.depth_tex.repeat_x = self.depth_tex.repeat_y = False
        self.resolved_fbo = ctx.framebuffer(color_attachments=[self.color_tex], depth_attachment=self.depth_tex)

    def begin(self, clear_color=(0.1, 0.1, 0.1, 1.0)):
        """Binds the target and clears it, ready for the frame's scene."""
        self.fbo.use()
        self.fbo.clear(*clear_color, depth=1.0)

    def resolve(self):
        """Copies the scene drawn so far (colour and depth) into the single-sample textures."""
        self.ctx.copy_framebuffer(self.resolved_fbo, self.fbo)

    def present(self, screen):
        """Copies the finished frame's colour to `screen` (resolving multisampling if the screen
        has a different sample count)."""
        self.ctx.copy_framebuffer(screen, self.present_fbo)

    def release(self):
        for resource in (self.present_fbo, self.fbo, self.resolved_fbo, self.color_ms, self.depth_ms,
                         self.color_tex, self.depth_tex):
            try:
                resource.release()
            except Exception:
                pass
