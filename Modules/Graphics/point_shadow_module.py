import moderngl
import glm


class PointShadowMap:
    """
    Omnidirectional shadow map for a single point light.

    moderngl has no clean way to attach a real GL_TEXTURE_CUBE_MAP to a
    framebuffer for single-pass layered rendering, so this uses the same
    approach as CascadedShadowMap: six independent 2D depth textures, one
    per cube face, each its own 90-degree perspective frustum. The PBR
    shader picks the right face by the major axis of the direction from
    the light to the fragment, then samples it exactly like a normal
    perspective shadow map.

    NOTE on the static-geometry caching that used to live here: an
    earlier version baked static objects into a separate texture once,
    then used ctx.copy_framebuffer() to blit that cached depth into the
    "live" texture every frame before drawing dynamic objects on top, to
    avoid re-rendering static geometry into the shadow pass every frame.
    That blit relies on glBlitFramebuffer's depth-copy behavior, which
    has real GL-level constraints (e.g. requiring GL_NEAREST filtering)
    that turned out to be unreliable to get right without a live GL
    context to test against - lights ended up fully self-shadowed even
    after fixing the filtering issue, meaning something about that path
    was still wrong and couldn't be diagnosed further from documentation
    alone. So for now, correctness wins over that optimization: both
    static and dynamic objects are redrawn into these six faces every
    frame, exactly like the directional cascades already do. If this
    becomes a real performance bottleneck later, revisiting the caching
    approach is a good follow-up - just plan to verify it with an actual
    running GL context this time, e.g. reading back a texel after the
    blit to directly confirm the copy happened, rather than reasoning
    about it from docs.
    """

    # Order matters: must match get_cube_face() in the fragment shader.
    FACE_DIRECTIONS = [
        glm.vec3(1, 0, 0),
        glm.vec3(-1, 0, 0),
        glm.vec3(0, 1, 0),
        glm.vec3(0, -1, 0),
        glm.vec3(0, 0, 1),
        glm.vec3(0, 0, -1),
    ]

    FACE_UPS = [
        glm.vec3(0, -1, 0),
        glm.vec3(0, -1, 0),
        glm.vec3(0, 0, 1),
        glm.vec3(0, 0, -1),
        glm.vec3(0, -1, 0),
        glm.vec3(0, -1, 0),
    ]

    def __init__(self, ctx, resolution=1024, near=0.05, far=25.0):
        self.ctx = ctx
        self.resolution = resolution
        self.near = near
        self.far = far

        self.position = glm.vec3(0.0)
        self.light_mvps = [glm.mat4(1.0) for _ in range(6)]

        # What the PBR shader samples each frame. Named "live_" (rather
        # than just "textures"/"fbos") to keep the attribute names
        # compatible with pbr_shader.bind_point_lights, which reads
        # shadow_map.live_textures and shadow_map.light_mvps directly.
        self.live_textures = [
            ctx.depth_texture((resolution, resolution)) for _ in range(6)
        ]
        self.live_fbos = [
            ctx.framebuffer(depth_attachment=tex) for tex in self.live_textures
        ]

        for tex in self.live_textures:
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = tex.repeat_y = False

        self._recompute_matrices()

    # -----------------------------------------------------------
    # SETUP
    # -----------------------------------------------------------

    def set_position(self, position):
        """Move the light, recomputing its 6 view/projection matrices."""
        new_pos = glm.vec3(position)
        if glm.distance(new_pos, self.position) > 1e-5:
            self.position = new_pos
            self._recompute_matrices()

    def _recompute_matrices(self):
        proj = glm.perspective(glm.radians(90.0), 1.0, self.near, self.far)
        for i in range(6):
            view = glm.lookAt(
                self.position,
                self.position + self.FACE_DIRECTIONS[i],
                self.FACE_UPS[i],
            )
            self.light_mvps[i] = proj * view

    # -----------------------------------------------------------
    # RUNTIME
    # -----------------------------------------------------------

    def destroy(self):
        for fbo in self.live_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        for tex in self.live_textures:
            try:
                tex.release()
            except Exception:
                pass