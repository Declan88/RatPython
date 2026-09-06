import moderngl
import glm
import numpy as np


class CascadedShadowMap:
    def __init__(self, ctx, resolution=2048, cascade_count=3):
        self.ctx = ctx
        self.resolution = resolution
        self.cascade_count = cascade_count
        self.num_cascades = cascade_count
        self.near = 0.1
        self.far = 100.0

        self.depth_textures = [
            ctx.depth_texture((resolution, resolution)) for _ in range(cascade_count)
        ]
        for tex in self.depth_textures:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = tex.repeat_y = False

        self.framebuffers = [
            ctx.framebuffer(depth_attachment=tex) for tex in self.depth_textures
        ]
        self.fbos = self.framebuffers  # Alias for compatibility
        self.light_mvps = [glm.mat4(1.0) for _ in range(cascade_count)]
        self.splits = [0.0 for _ in range(max(0, cascade_count - 1))]

        self.program = ctx.program(
            vertex_shader="""
            #version 330
            uniform mat4 u_light_mvp;
            in vec3 in_position;
            void main() {
                gl_Position = u_light_mvp * vec4(in_position, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            void main() {}
            """,
        )

    def _get_camera_basis(self, camera):
        inv_view = glm.inverse(camera.get_view_matrix())
        return (
            glm.vec3(inv_view[3]),
            glm.normalize(glm.vec3(inv_view[0])),
            glm.normalize(glm.vec3(inv_view[1])),
            glm.normalize(-glm.vec3(inv_view[2])),
        )

    def _get_frustum_corners(self, camera, near_d, far_d):
        proj = camera.get_projection_matrix()
        pos, right, up, forward = self._get_camera_basis(camera)
        px, py = float(proj[0][0]), float(proj[1][1])
        fov_y = 2.0 * np.arctan(1.0 / py) if abs(px) >= 1e-6 else np.radians(60.0)
        aspect = (py / px) if abs(px) >= 1e-6 else 1.6

        tan_half = np.tan(fov_y * 0.5)
        nh, nw = near_d * tan_half, near_d * tan_half * aspect
        fh, fw = far_d * tan_half, far_d * tan_half * aspect
        nc, fc = pos + forward * near_d, pos + forward * far_d

        return [
            nc - right * nw - up * nh,
            nc + right * nw - up * nh,
            nc + right * nw + up * nh,
            nc - right * nw + up * nh,
            fc - right * fw - up * fh,
            fc + right * fw - up * fh,
            fc + right * fw + up * fh,
            fc - right * fw + up * fh,
        ]

    def update(self, camera, light_dir):
        self.near = max(float(getattr(camera, "near", 0.1)), 0.01)
        self.far = max(float(getattr(camera, "far", 100.0)), self.near + 1.0)

        lambda_val = 0.75
        cascade_splits = []
        for i in range(self.cascade_count):
            p = (i + 1) / float(self.cascade_count)
            log_s = self.near * (self.far / self.near) ** p
            uni_s = self.near + (self.far - self.near) * p
            cascade_splits.append(lambda_val * log_s + (1.0 - lambda_val) * uni_s)

        self.splits = cascade_splits[:-1]

        light = glm.normalize(
            glm.vec3(light_dir)
            if isinstance(light_dir, np.ndarray)
            else glm.vec3(light_dir.x, light_dir.y, light_dir.z)
        )
        if glm.length(light) < 1e-6:
            light = glm.vec3(0.5, 1.0, 0.8)

        world_up = (
            glm.vec3(0.0, 0.0, 1.0)
            if abs(glm.dot(light, glm.vec3(0, 1, 0))) > 0.95
            else glm.vec3(0, 1, 0)
        )
        prev_split = self.near

        for i in range(self.cascade_count):
            curr_split = cascade_splits[i]
            corners = self._get_frustum_corners(camera, prev_split, curr_split)
            center = sum(corners, glm.vec3(0.0)) / float(len(corners))

            light_view = glm.lookAt(
                center + light * max(curr_split * 2.0, 50.0), center, world_up
            )
            min_xyz = glm.vec3(float("inf"))
            max_xyz = glm.vec3(float("-inf"))

            for corner in corners:
                pt = light_view * glm.vec4(corner, 1.0)
                min_xyz = glm.min(min_xyz, glm.vec3(pt))
                max_xyz = glm.max(max_xyz, glm.vec3(pt))

            xy_pad = max(1.0, (max_xyz.x - min_xyz.x) * 0.02)
            near_p = max(0.01, -max_xyz.z - 50.0)
            far_p = -min_xyz.z + 50.0

            light_proj = glm.ortho(
                min_xyz.x - xy_pad,
                max_xyz.x + xy_pad,
                min_xyz.y - xy_pad,
                max_xyz.y + xy_pad,
                near_p,
                far_p,
            )
            self.light_mvps[i] = light_proj * light_view
            prev_split = curr_split

    def render(self, render_callback):
        for i in range(self.cascade_count):
            fbo = self.framebuffers[i]
            fbo.use()
            fbo.clear(depth=1.0)
            self.program["u_light_mvp"].write(self.light_mvps[i])
            render_callback(self.program)

    def destroy(self):
        for fbo in self.framebuffers:
            try:
                fbo.release()
            except Exception:
                pass
        for tex in self.depth_textures:
            try:
                tex.release()
            except Exception:
                pass
        try:
            self.program.release()
        except Exception:
            pass
