"""
The offscreen render target every scene draws into (the window itself has no MSAA): a multisampled colour + depth,
resolved into the window at the end of the frame (present) and, on frames that need it, into single-sample copies
mid-frame for screen-space reflections (resolve).

Both resolves are a fullscreen-triangle draw that averages the samples (texelFetch on a sampler2DMS), not
glBlitFramebuffer: with NVIDIA's threaded OpenGL driver a blit made the game thread wait for the driver's worker to
drain its whole queue - twice a frame, ~0.6 ms of a ~2 ms profiled frame - where a draw is simply queued. A shader
also doesn't care that the window's colour format differs from the target's (which strict drivers reject for a blit).
"""

import moderngl

# Texture units of the resolve's own two samplers: above pbr_shader's 0-15, so a mid-frame resolve can't disturb
# a frame-wide binding (shadow maps, reflection environment).
_COLOR_UNIT = 16
_DEPTH_UNIT = 17

_VERT = """
#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);     // full-screen triangle, no vertex buffer
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# MS: the source is multisampled (sampler2DMS, averaged) or not (sampler2D, one texel). DEPTH: also writes depth.
_FRAG = """
#version 330
#if MS
uniform sampler2DMS u_color;
#if DEPTH
uniform sampler2DMS u_depth;
#endif
uniform int u_samples;
#else
uniform sampler2D u_color;
#if DEPTH
uniform sampler2D u_depth;
#endif
#endif
out vec4 frag_color;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
#if MS
    vec4 sum = vec4(0.0);
    for (int i = 0; i < u_samples; ++i) sum += texelFetch(u_color, p, i);
    frag_color = sum / float(u_samples);
#if DEPTH
    gl_FragDepth = texelFetch(u_depth, p, 0).r;     // like a blit's depth resolve: one sample, not an average
#endif
#else
    frag_color = texelFetch(u_color, p, 0);
#if DEPTH
    gl_FragDepth = texelFetch(u_depth, p, 0).r;
#endif
#endif
}
"""


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

        # Single-sample copies for SSR (filled by resolve(), only on frames a reflective surface is drawn).
        self.color_tex = ctx.texture(self.size, 4)
        self.color_tex.repeat_x = self.color_tex.repeat_y = False
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_tex = ctx.depth_texture(self.size)
        self.depth_tex.repeat_x = self.depth_tex.repeat_y = False
        self.resolved_fbo = ctx.framebuffer(color_attachments=[self.color_tex], depth_attachment=self.depth_tex)

        defines = f"#define MS {1 if self.samples else 0}\n"
        self._present_prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG.replace(
            "#version 330", "#version 330\n" + defines + "#define DEPTH 0"))
        self._resolve_prog = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG.replace(
            "#version 330", "#version 330\n" + defines + "#define DEPTH 1"))
        for prog in (self._present_prog, self._resolve_prog):
            prog["u_color"].value = _COLOR_UNIT
            if "u_depth" in prog:
                prog["u_depth"].value = _DEPTH_UNIT
            if "u_samples" in prog:
                prog["u_samples"].value = max(1, self.samples)
        self._present_vao = ctx.vertex_array(self._present_prog, [])
        self._resolve_vao = ctx.vertex_array(self._resolve_prog, [])

    def begin(self, clear_color=(0.1, 0.1, 0.1, 1.0)):
        """Binds the target and clears it, ready for the frame's scene."""
        self.fbo.use()
        self.fbo.clear(*clear_color, depth=1.0)

    def _draw(self, vao, depth):
        ctx = self.ctx
        ctx.disable(moderngl.BLEND | moderngl.CULL_FACE)
        if depth:
            ctx.enable(moderngl.DEPTH_TEST)
            ctx.depth_func = "1"            # always: every pixel's depth is written
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        vao.render(moderngl.TRIANGLES, vertices=3)
        # Back to the scene's defaults (what the passes after this expect).
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.depth_func = "<"

    def resolve(self):
        """Copies the scene drawn so far (colour and depth) into the single-sample textures, then rebinds the target."""
        self.resolved_fbo.use()
        self.color_ms.use(location=_COLOR_UNIT)
        self.depth_ms.use(location=_DEPTH_UNIT)
        self._draw(self._resolve_vao, depth=True)
        self.fbo.use()

    def present(self, screen):
        """Resolves the finished frame's colour into `screen`."""
        screen.use()
        self.color_ms.use(location=_COLOR_UNIT)
        self._draw(self._present_vao, depth=False)

    def release(self):
        for resource in (self._present_vao, self._resolve_vao, self._present_prog, self._resolve_prog,
                         self.fbo, self.resolved_fbo, self.color_ms, self.depth_ms, self.color_tex, self.depth_tex):
            try:
                resource.release()
            except Exception:
                pass
