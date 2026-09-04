from __future__ import annotations

import moderngl
import glm

from Modules.Graphics.pbr_shader import (
    create_program,
    bind_material
)
from Modules.Graphics.shadow_module import CascadedShadowMap
from Modules.Graphics.model_loader import load_glb


class Scene:
    def __init__(self, ctx):
        self.ctx = ctx

        self.static_objects = []
        self.dynamic_objects = []

        self.light_dir = glm.vec3(
            0.5,
            1.0,
            0.8
        )

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"

        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        self.pbr_program = create_program(self.ctx)

        self.shadow_program = self.ctx.program(
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

        self.shadow_manager = CascadedShadowMap(
            self.ctx
        )

    # =============================================================
    # OBJECT LOADING
    # =============================================================

    def _load_object(self, model_path):
        model = load_glb(
            model_path,
            self.ctx,
            self.pbr_program
        )

        if model is None:
            return None

        shadow_model = load_glb(
            model_path,
            self.ctx,
            self.shadow_program
        )

        if shadow_model is None:
            try:
                model["vao"].release()
            except Exception:
                pass

            texture = model.get("texture")

            if texture is not None:
                try:
                    texture.release()
                except Exception:
                    pass

            mr_texture = model.get(
                "metallic_roughness_texture"
            )

            if mr_texture is not None:
                try:
                    mr_texture.release()
                except Exception:
                    pass

            return None

        return {
            "vao": model["vao"],
            "shadow_vao": shadow_model["vao"],

            "texture": model.get("texture"),

            "metallic_roughness_texture":
                model.get(
                    "metallic_roughness_texture"
                ),

            "metallic": model.get(
                "metallic",
                0.1
            ),

            "roughness": model.get(
                "roughness",
                0.5
            ),

            "emissive": model.get(
                "emissive",
                [0.0, 0.0, 0.0]
            ),

            "has_texture": model.get(
                "has_texture",
                0
            ),

            "has_metallic_roughness_texture":
                model.get(
                    "has_metallic_roughness_texture",
                    0
                )
        }

    # =============================================================
    # STATIC OBJECTS
    # =============================================================

    def add_static(
        self,
        model_path,
        position=None,
        rotation=None,
        scale=None,
        transform=None,
        metallic=None,
        roughness=None
    ):
        model = self._load_object(
            model_path
        )

        if model is None:
            return None

        if metallic is not None:
            model["metallic"] = metallic

        if roughness is not None:
            model["roughness"] = roughness

        if transform is not None:
            model["transform"] = glm.mat4(
                transform
            )
        else:
            if position is None:
                position = glm.vec3(0.0)

            if rotation is None:
                rotation = glm.vec3(0.0)

            if scale is None:
                scale = glm.vec3(1.0)

            model["position"] = glm.vec3(
                position
            )

            model["rotation"] = glm.vec3(
                rotation
            )

            model["scale"] = glm.vec3(
                scale
            )

        self.static_objects.append(
            model
        )

        return model

    # =============================================================
    # DYNAMIC OBJECTS
    # =============================================================

    def add_dynamic(
        self,
        model_path,
        position=glm.vec3(0.0),
        rotation=glm.vec3(0.0),
        scale=glm.vec3(1.0),
        rot_speed=0.0,
        transform=None,
        metallic=None,
        roughness=None
    ):
        model = self._load_object(
            model_path
        )

        if model is None:
            return None

        if metallic is not None:
            model["metallic"] = metallic

        if roughness is not None:
            model["roughness"] = roughness

        if transform is not None:
            model["transform"] = glm.mat4(
                transform
            )
        else:
            model["position"] = glm.vec3(
                position
            )

            model["rotation"] = glm.vec3(
                rotation
            )

            model["scale"] = glm.vec3(
                scale
            )

        model["rot_speed"] = float(
            rot_speed
        )

        self.dynamic_objects.append(
            model
        )

        return model

    # =============================================================
    # MODEL MATRIX
    # =============================================================

    def _get_model_matrix(self, obj):
        if "transform" in obj:
            return glm.mat4(
                obj["transform"]
            )

        model = glm.mat4(1.0)

        model = glm.translate(
            model,
            obj["position"]
        )

        rotation = obj["rotation"]

        model = glm.rotate(
            model,
            rotation.x,
            glm.vec3(1.0, 0.0, 0.0)
        )

        model = glm.rotate(
            model,
            rotation.y,
            glm.vec3(0.0, 1.0, 0.0)
        )

        model = glm.rotate(
            model,
            rotation.z,
            glm.vec3(0.0, 0.0, 1.0)
        )

        model = glm.scale(
            model,
            obj["scale"]
        )

        return model

    # =============================================================
    # UPDATE
    # =============================================================

    def update(self, dt):
        for obj in self.dynamic_objects:
            rot_speed = obj.get(
                "rot_speed",
                0.0
            )

            if rot_speed != 0.0:
                if "rotation" in obj:
                    obj["rotation"].y += (
                        rot_speed * dt
                    )

    # =============================================================
    # SHADOW PASS
    # =============================================================

    def _render_shadows(self, camera):
        self.shadow_manager.update(
            camera,
            self.light_dir
        )

        resolution = (
            self.shadow_manager.resolution
        )

        self.ctx.enable(
            moderngl.DEPTH_TEST
        )

        self.ctx.depth_func = "<="

        self.ctx.enable(
            moderngl.CULL_FACE
        )

        self.ctx.cull_face = "front"

        old_viewport = self.ctx.viewport

        for cascade in range(
            self.shadow_manager.num_cascades
        ):
            framebuffer = (
                self.shadow_manager.framebuffers[
                    cascade
                ]
            )

            light_vp = (
                self.shadow_manager.light_mvps[
                    cascade
                ]
            )

            framebuffer.use()

            self.ctx.viewport = (
                0,
                0,
                resolution,
                resolution
            )

            framebuffer.clear(
                depth=1.0
            )

            for obj in self.static_objects:
                model_matrix = (
                    self._get_model_matrix(obj)
                )

                light_mvp = (
                    light_vp *
                    model_matrix
                )

                self.shadow_program[
                    "u_light_mvp"
                ].write(
                    light_mvp.to_bytes()
                )

                obj["shadow_vao"].render()

            for obj in self.dynamic_objects:
                model_matrix = (
                    self._get_model_matrix(obj)
                )

                light_mvp = (
                    light_vp *
                    model_matrix
                )

                self.shadow_program[
                    "u_light_mvp"
                ].write(
                    light_mvp.to_bytes()
                )

                obj["shadow_vao"].render()

        self.ctx.screen.use()

        self.ctx.viewport = old_viewport

        self.ctx.enable(
            moderngl.DEPTH_TEST
        )

        self.ctx.depth_func = "<"

        self.ctx.enable(
            moderngl.CULL_FACE
        )

        self.ctx.cull_face = "back"

    # =============================================================
    # PBR PASS
    # =============================================================

    def _render_scene(self, camera):
        self.ctx.screen.use()

        self.ctx.enable(
            moderngl.DEPTH_TEST
        )

        self.ctx.depth_func = "<"

        self.ctx.enable(
            moderngl.CULL_FACE
        )

        self.ctx.cull_face = "back"

        for obj in self.static_objects:
            model_matrix = (
                self._get_model_matrix(obj)
            )

            bind_material(
                self.pbr_program,
                obj,
                model_matrix,
                camera,
                self.light_dir,
                self.shadow_manager
            )

            obj["vao"].render()

        for obj in self.dynamic_objects:
            model_matrix = (
                self._get_model_matrix(obj)
            )

            bind_material(
                self.pbr_program,
                obj,
                model_matrix,
                camera,
                self.light_dir,
                self.shadow_manager
            )

            obj["vao"].render()

    # =============================================================
    # RENDER
    # =============================================================

    def render(self, camera, prog=None):
        self._render_shadows(
            camera
        )

        self._render_scene(
            camera
        )

    # =============================================================
    # DESTROY
    # =============================================================

    def destroy(self):
        for obj in self.static_objects:
            self._release_object(obj)

        for obj in self.dynamic_objects:
            self._release_object(obj)

        if self.shadow_manager is not None:
            try:
                self.shadow_manager.destroy()
            except Exception:
                pass

            self.shadow_manager = None

        if self.pbr_program is not None:
            try:
                self.pbr_program.release()
            except Exception:
                pass

            self.pbr_program = None

        if self.shadow_program is not None:
            try:
                self.shadow_program.release()
            except Exception:
                pass

            self.shadow_program = None

        self.static_objects.clear()
        self.dynamic_objects.clear()

    # =============================================================
    # RELEASE OBJECT
    # =============================================================

    def _release_object(self, obj):
        vao = obj.get("vao")

        if vao is not None:
            try:
                vao.release()
            except Exception:
                pass

        shadow_vao = obj.get(
            "shadow_vao"
        )

        if shadow_vao is not None:
            try:
                shadow_vao.release()
            except Exception:
                pass

        texture = obj.get(
            "texture"
        )

        if texture is not None:
            try:
                texture.release()
            except Exception:
                pass

        mr_texture = obj.get(
            "metallic_roughness_texture"
        )

        if mr_texture is not None:
            try:
                mr_texture.release()
            except Exception:
                pass