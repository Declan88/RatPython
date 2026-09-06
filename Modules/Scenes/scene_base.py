from __future__ import annotations

import moderngl
import glm

from Modules.Graphics.pbr_shader import (
    create_program,
    bind_material,
    bind_point_lights,
    MAX_SHADOW_POINT_LIGHTS
)
from Modules.Graphics.shadow_module import CascadedShadowMap
from Modules.Graphics.point_shadow_module import PointShadowMap
from Modules.Graphics.gltf_lights import extract_punctual_lights
from Modules.Graphics.model_loader import load_glb


class Scene:
    def __init__(self, ctx):
        self.ctx = ctx

        self.static_objects = []
        self.dynamic_objects = []

        self.point_lights = []
        self.point_shadow_maps = []

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

        # New static geometry invalidates any cached point-light
        # shadow bakes, since they only cover static objects.
        self.mark_static_dirty()

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
    # POINT LIGHTS
    # =============================================================

    def add_point_light(
        self,
        position,
        color=(1.0, 1.0, 1.0),
        intensity=1.0,
        radius=10.0,
        cast_shadows=False
    ):
        light = {
            "position": glm.vec3(position),
            "color": glm.vec3(color),
            "intensity": float(intensity),
            "radius": float(radius),
            "shadow_map": None
        }

        if cast_shadows:
            if len(self.point_shadow_maps) < MAX_SHADOW_POINT_LIGHTS:
                shadow_map = PointShadowMap(
                    self.ctx,
                    resolution=1024,
                    near=0.05,
                    far=max(radius * 2.0, 1.0)
                )

                shadow_map.set_position(
                    light["position"]
                )

                self.point_shadow_maps.append(
                    shadow_map
                )

                light["shadow_map"] = shadow_map
            else:
                print(
                    f"[Scene] Max shadow-casting point lights "
                    f"({MAX_SHADOW_POINT_LIGHTS}) reached; light at "
                    f"{tuple(light['position'])} will be unshadowed."
                )

        self.point_lights.append(
            light
        )

        return light

    def add_lights_from_glb(
        self,
        model_path,
        cast_shadows=False,
        default_radius=8.0,
        intensity_multiplier=1.0,
        radius_multiplier=1.0
    ):
        """Reads KHR_lights_punctual lights out of a glb and adds any
        point lights found as real point lights in the scene.

        intensity_multiplier: scales the converted intensity (see the
        unit-conversion note in gltf_lights.extract_punctual_lights) -
        useful since the candela->linear conversion is only approximate
        and glTF-authored lights often end up too dim/bright as-is.

        radius_multiplier: scales the light's falloff radius (from the
        glTF "range" field, or default_radius if range wasn't authored).
        Useful because "range" is just a culling hint in the glTF spec,
        not a value tuned for your shader's specific falloff curve.
        """
        added = []

        for light in extract_punctual_lights(model_path):
            if light["type"] != "point":
                print(
                    f"[Scene] Skipping '{light['type']}' light from "
                    f"{model_path} - only 'point' lights are supported "
                    f"(no spot cone / directional handling)."
                )
                continue

            added.append(
                self.add_point_light(
                    position=light["position"],
                    color=light["color"],
                    intensity=(light["intensity"] or 1.0) * intensity_multiplier,
                    radius=(light["range"] or default_radius) * radius_multiplier,
                    cast_shadows=cast_shadows,
                )
            )

        return added

    def mark_static_dirty(self):
        """No-op for now. Point-light shadows used to cache static
        geometry and only re-draw dynamic objects per frame, which
        this method invalidated - that caching was removed (see
        PointShadowMap's docstring) because the depth-copy step it
        relied on couldn't be verified to work correctly. Kept here so
        existing call sites like add_static() don't break; safe to
        remove if caching is reintroduced with a different mechanism
        or dropped from the API entirely."""
        pass

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
    # DIRECTIONAL SHADOW PASS
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
    # POINT LIGHT SHADOW PASS
    # =============================================================

    def _render_point_shadows(self):
        if not self.point_shadow_maps:
            return

        old_viewport = self.ctx.viewport

        self.ctx.enable(
            moderngl.DEPTH_TEST
        )

        self.ctx.depth_func = "<="

        self.ctx.enable(
            moderngl.CULL_FACE
        )

        self.ctx.cull_face = "front"

        for shadow_map in self.point_shadow_maps:
            resolution = shadow_map.resolution

            for face in range(6):
                framebuffer = shadow_map.live_fbos[face]

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

                light_vp = shadow_map.light_mvps[face]

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

        # Point-light data and shadow textures are identical for every
        # object this frame, so bind them once here rather than inside
        # the per-object loop below.
        bind_point_lights(
            self.pbr_program,
            self.point_lights
        )

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

        self._render_point_shadows()

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

        for shadow_map in self.point_shadow_maps:
            try:
                shadow_map.destroy()
            except Exception:
                pass

        self.point_shadow_maps.clear()
        self.point_lights.clear()

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