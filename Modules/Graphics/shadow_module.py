import moderngl
import glm
import numpy as np


class CascadedShadowMap:

    def __init__(
        self,
        ctx,
        resolution=2048,
        cascade_count=3
    ):
        self.ctx = ctx
        self.resolution = resolution

        self.cascade_count = cascade_count
        self.num_cascades = cascade_count

        self.depth_textures = []
        self.fbos = []

        self.framebuffers = self.fbos

        for _ in range(cascade_count):
            tex = ctx.depth_texture(
                (
                    resolution,
                    resolution
                )
            )

            tex.filter = (
                moderngl.NEAREST,
                moderngl.NEAREST
            )

            tex.repeat_x = False
            tex.repeat_y = False

            self.depth_textures.append(tex)

            self.fbos.append(
                ctx.framebuffer(
                    depth_attachment=tex
                )
            )

        self.light_mvps = [
            glm.mat4(1.0)
            for _ in range(cascade_count)
        ]

        self.splits = [
            0.0
            for _ in range(
                max(0, cascade_count - 1)
            )
        ]

        self.near = 0.1
        self.far = 100.0

        self.program = ctx.program(
            vertex_shader="""
            #version 330

            uniform mat4 u_light_mvp;

            in vec3 in_position;

            void main()
            {
                gl_Position =
                    u_light_mvp *
                    vec4(in_position, 1.0);
            }
            """,

            fragment_shader="""
            #version 330

            void main()
            {
            }
            """
        )

    # ---------------------------------------------------------
    # CAMERA BASIS
    # ---------------------------------------------------------

    def _get_camera_basis(self, camera):

        view = camera.get_view_matrix()
        inv_view = glm.inverse(view)

        position = glm.vec3(
            inv_view[3][0],
            inv_view[3][1],
            inv_view[3][2]
        )

        right = glm.normalize(
            glm.vec3(
                inv_view[0][0],
                inv_view[0][1],
                inv_view[0][2]
            )
        )

        up = glm.normalize(
            glm.vec3(
                inv_view[1][0],
                inv_view[1][1],
                inv_view[1][2]
            )
        )

        forward = glm.normalize(
            glm.vec3(
                -inv_view[2][0],
                -inv_view[2][1],
                -inv_view[2][2]
            )
        )

        return (
            position,
            right,
            up,
            forward
        )

    # ---------------------------------------------------------
    # FRUSTUM CORNERS
    # ---------------------------------------------------------

    def _get_frustum_corners(
        self,
        camera,
        near_distance,
        far_distance
    ):
        projection = camera.get_projection_matrix()

        (
            position,
            right,
            up,
            forward
        ) = self._get_camera_basis(camera)

        px = float(
            projection[0][0]
        )

        py = float(
            projection[1][1]
        )

        if abs(px) < 0.000001:

            aspect = 16.0 / 9.0
            fov_y = np.radians(60.0)

        else:

            aspect = py / px

            fov_y = (
                2.0 *
                np.arctan(
                    1.0 / py
                )
            )

        tan_half_fov = np.tan(
            fov_y * 0.5
        )

        near_height = (
            near_distance *
            tan_half_fov
        )

        near_width = (
            near_height *
            aspect
        )

        far_height = (
            far_distance *
            tan_half_fov
        )

        far_width = (
            far_height *
            aspect
        )

        near_center = (
            position +
            forward * near_distance
        )

        far_center = (
            position +
            forward * far_distance
        )

        return [

            near_center
            - right * near_width
            - up * near_height,

            near_center
            + right * near_width
            - up * near_height,

            near_center
            + right * near_width
            + up * near_height,

            near_center
            - right * near_width
            + up * near_height,

            far_center
            - right * far_width
            - up * far_height,

            far_center
            + right * far_width
            - up * far_height,

            far_center
            + right * far_width
            + up * far_height,

            far_center
            - right * far_width
            + up * far_height
        ]

    # ---------------------------------------------------------
    # LIGHT UP VECTOR
    # ---------------------------------------------------------

    def _get_light_up(self, direction):

        world_up = glm.vec3(
            0.0,
            1.0,
            0.0
        )

        if abs(
            glm.dot(
                direction,
                world_up
            )
        ) > 0.95:

            return glm.vec3(
                0.0,
                0.0,
                1.0
            )

        return world_up

    # ---------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------

    def update(
        self,
        camera,
        light_dir
    ):

        self.near = float(
            getattr(
                camera,
                "near",
                0.1
            )
        )

        self.far = float(
            getattr(
                camera,
                "far",
                100.0
            )
        )

        self.near = max(
            self.near,
            0.01
        )

        self.far = max(
            self.far,
            self.near + 1.0
        )

        # -----------------------------------------------------
        # CASCADE SPLITS
        # -----------------------------------------------------

        lambda_value = 0.75

        cascade_splits = []

        for i in range(
            self.cascade_count
        ):

            p = (
                (i + 1)
                / float(self.cascade_count)
            )

            logarithmic = (
                self.near *
                (
                    self.far /
                    self.near
                ) ** p
            )

            uniform = (
                self.near +
                (
                    self.far -
                    self.near
                ) * p
            )

            split = (
                lambda_value *
                logarithmic
                +
                (1.0 - lambda_value) *
                uniform
            )

            cascade_splits.append(
                split
            )

        self.splits = (
            cascade_splits[:-1]
        )

        # -----------------------------------------------------
        # LIGHT DIRECTION
        # -----------------------------------------------------

        if isinstance(
            light_dir,
            np.ndarray
        ):

            light = glm.vec3(
                float(light_dir[0]),
                float(light_dir[1]),
                float(light_dir[2])
            )

        else:

            light = glm.vec3(
                float(light_dir.x),
                float(light_dir.y),
                float(light_dir.z)
            )

        if glm.length(light) < 0.000001:

            light = glm.vec3(
                0.5,
                1.0,
                0.8
            )

        light = glm.normalize(
            light
        )

        # -----------------------------------------------------
        # BUILD CASCADES
        # -----------------------------------------------------

        previous_split = self.near

        for i in range(
            self.cascade_count
        ):

            current_split = (
                cascade_splits[i]
            )

            corners = (
                self._get_frustum_corners(
                    camera,
                    previous_split,
                    current_split
                )
            )

            # -------------------------------------------------
            # FRUSTUM CENTER
            # -------------------------------------------------

            center = glm.vec3(
                0.0
            )

            for corner in corners:
                center += corner

            center /= float(
                len(corners)
            )

            # -------------------------------------------------
            # LIGHT CAMERA
            # -------------------------------------------------

            light_distance = max(
                current_split * 2.0,
                50.0
            )

            light_position = (
                center +
                light * light_distance
            )

            light_view = glm.lookAt(
                light_position,
                center,
                self._get_light_up(light)
            )

            # -------------------------------------------------
            # LIGHT-SPACE BOUNDS
            # -------------------------------------------------

            min_x = float("inf")
            max_x = float("-inf")

            min_y = float("inf")
            max_y = float("-inf")

            min_z = float("inf")
            max_z = float("-inf")

            for corner in corners:

                point = (
                    light_view *
                    glm.vec4(
                        corner,
                        1.0
                    )
                )

                min_x = min(
                    min_x,
                    float(point.x)
                )

                max_x = max(
                    max_x,
                    float(point.x)
                )

                min_y = min(
                    min_y,
                    float(point.y)
                )

                max_y = max(
                    max_y,
                    float(point.y)
                )

                min_z = min(
                    min_z,
                    float(point.z)
                )

                max_z = max(
                    max_z,
                    float(point.z)
                )

            # -------------------------------------------------
            # XY PADDING
            # -------------------------------------------------

            width = (
                max_x - min_x
            )

            height = (
                max_y - min_y
            )

            xy_padding = max(
                1.0,
                width * 0.02,
                height * 0.02
            )

            min_x -= xy_padding
            max_x += xy_padding

            min_y -= xy_padding
            max_y += xy_padding

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Light-space Z is negative because the light
            # camera looks down -Z.
            #
            # glm.ortho() expects positive near/far DISTANCES.
            #
            # Convert:
            #
            #   max_z -> nearest distance
            #   min_z -> farthest distance
            #
            # -------------------------------------------------

            near_plane = max(
                0.01,
                -max_z
            )

            far_plane = max(
                near_plane + 1.0,
                -min_z
            )

            # Extra depth padding so shadow casters outside
            # the camera frustum still have room to appear.
            near_plane = max(
                0.01,
                near_plane - 50.0
            )

            far_plane += 50.0

            # -------------------------------------------------
            # ORTHOGRAPHIC LIGHT PROJECTION
            # -------------------------------------------------

            light_projection = glm.ortho(
                min_x,
                max_x,
                min_y,
                max_y,
                near_plane,
                far_plane
            )

            self.light_mvps[i] = (
                light_projection *
                light_view
            )

            previous_split = (
                current_split
            )

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------

    def destroy(self):

        for fbo in self.fbos:

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