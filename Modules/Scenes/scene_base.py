from __future__ import annotations

from pathlib import Path

import moderngl
import glm
import numpy as np

from Modules.Graphics.pbr_shader import (
    create_program,
    bind_material,
    bind_point_lights
)
from Modules.Graphics.shadow_module import CascadedShadowMap
from Modules.Graphics.point_shadow_module import PointShadowMap
from Modules.Graphics.gltf_lights import extract_punctual_lights
from Modules.Graphics.lightmap_baker import (
    create_bake_program,
    create_lightmap,
    bake_point_light
)
from Modules.Graphics.model_loader import load_glb


class Scene:
    def __init__(self, ctx, recalculate_shadows=True):
        self.ctx = ctx

        # If True, bake_static_lighting() recomputes lighting from
        # scratch and saves it to disk. If False, it loads the
        # previously-saved lightmaps instead of re-baking, which is
        # much faster - useful once you're happy with the lighting and
        # don't want to pay the bake cost on every launch. Falls back
        # to baking automatically if the cached files aren't there yet
        # (e.g. first run).
        self.recalculate_shadows = recalculate_shadows
        self.lightmap_dir = Path("Assets/Lightmaps") / self.__class__.__name__

        self.static_objects = []
        self.dynamic_objects = []

        self.point_lights = []

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

        self.bake_program = create_bake_program(
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

        # Only bother with a bake-program VAO if the glb actually has a
        # second UV channel to bake into - otherwise this is wasted work.
        lightmap_model = load_glb(
            model_path,
            self.ctx,
            self.bake_program
        ) if model.get("has_lightmap_uv") else None

        if shadow_model is None:
            try:
                model["vao"].release()
            except Exception:
                pass

            if lightmap_model is not None:
                try:
                    lightmap_model["vao"].release()
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
            "lightmap_vao": lightmap_model["vao"] if lightmap_model else None,
            "has_lightmap_uv": model.get("has_lightmap_uv", False),
            "lightmap_texture": None,

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
        """cast_shadows now means "shadow-test this light against static
        geometry during bake_static_lighting()", not "give it a real-time
        cubemap" - point lights are always unshadowed in real time (see
        pbr_shader.py's docstring). No cap on how many lights can request
        this: baking processes lights one at a time, so it doesn't hit
        the shader register-limit issue a large real-time array would."""
        light = {
            "position": glm.vec3(position),
            "color": glm.vec3(color),
            "intensity": float(intensity),
            "radius": float(radius),
            "bake_shadows": bool(cast_shadows)
        }

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
        """No-op. Kept only so existing call sites like add_static()
        don't break. There's no runtime point-light shadow cache to
        invalidate anymore - point lights are always unshadowed in
        real time, and baked shadows are recomputed by explicitly
        calling bake_static_lighting() again, not by a dirty flag."""
        pass

    # =============================================================
    # LIGHTMAP BAKING
    # =============================================================

    def bake_static_lighting(self, lightmap_resolution=256):
        """Call this once, after adding all static objects and point
        lights, to bake shadow-tested point light contributions (from
        lights added with cast_shadows=True) into each static object's
        lightmap. Lights are baked one at a time (additively blended) -
        see lightmap_baker.py's docstring for why that matters.

        The directional light is deliberately NEVER baked here - it
        stays fully real-time via CascadedShadowMap for every object,
        static or dynamic (see _render_shadows). Baking it too would
        double-count it: the runtime shader already adds a real-time,
        correctly-shadowed directional term for every object regardless
        of whether it has a lightmap, so an object with both a baked
        directional contribution AND the real-time one would show it
        twice, at roughly double brightness.

        Point lights added with cast_shadows=False are not baked at all;
        they stay real-time-unshadowed only (see add_point_light)."""
        eligible = [
            obj for obj in self.static_objects
            if obj.get("has_lightmap_uv") and obj.get("lightmap_vao") is not None
        ]

        # NOTE: this runs once, typically from a Scene subclass's
        # __init__ - i.e. before the app's main loop has necessarily set
        # a real window viewport. Capturing/restoring self.ctx.viewport
        # like the per-frame render passes do isn't safe here, since
        # that snapshot could just be moderngl's early default rather
        # than the actual window size - restore to the real framebuffer
        # size explicitly instead.
        restore_viewport = (0, 0, *self.ctx.screen.size)

        if not eligible:
            print(
                f"[Scene] bake_static_lighting: no static objects have lightmap UVs - "
                f"nothing to bake ({len(self.static_objects)} static object(s) checked)."
            )
            for i, obj in enumerate(self.static_objects):
                print(f"  static_objects[{i}]: has_lightmap_uv={obj.get('has_lightmap_uv')}")
            return

        self.lightmap_dir.mkdir(parents=True, exist_ok=True)
        cache_paths = [self.lightmap_dir / f"lightmap_{i}.npy" for i in range(len(eligible))]

        if not self.recalculate_shadows and all(p.exists() for p in cache_paths):
            for obj, path in zip(eligible, cache_paths):
                if obj.get("lightmap_texture") is not None:
                    obj["lightmap_texture"].release()

                # .astype/.tobytes() rather than relying on the loaded
                # array's dtype directly - guards against a cache file
                # ever ending up float32 (e.g. from a different numpy
                # version) not matching the f2 (half-float) texture
                # format below.
                array = np.load(path).astype(np.float16)
                texture = self.ctx.texture((array.shape[1], array.shape[0]), 4, array.tobytes(), dtype="f2")
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                obj["lightmap_texture"] = texture

            return

        if not self.recalculate_shadows:
            print(f"[Scene] recalculate_shadows is False but cached lightmaps are missing/incomplete in {self.lightmap_dir} - baking instead.")

        print(f"[Scene] Baking lighting for {len(eligible)} static object(s)...")

        for obj in eligible:
            if obj.get("lightmap_texture") is not None:
                obj["lightmap_texture"].release()
            obj["lightmap_texture"] = create_lightmap(self.ctx, lightmap_resolution)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE

        # --- Point lights flagged for baked shadows, one at a time.
        # Each gets its own temporary 6-face shadow cube (static
        # geometry only), used just for this bake, then destroyed. ---
        for light in self.point_lights:
            if not light.get("bake_shadows"):
                continue

            temp_shadow = PointShadowMap(
                self.ctx, resolution=1024, near=0.05, far=max(light["radius"] * 2.0, 1.0)
            )
            temp_shadow.set_position(light["position"])

            # No culling for this depth pass. Front-face culling only
            # gives correct results for closed/watertight geometry (it
            # relies on a "back" face still being there to record depth
            # once the front is culled) - for thin, single-sided walls
            # (a flat quad with no backside), culling the front face on
            # certain sides of the light can remove the wall from this
            # pass ENTIRELY, leaving zero depth data recorded there. No
            # depth data means the shadow test can never find an
            # occluder, so light passes straight through as if the wall
            # weren't there - which is exactly "bleeding through walls"
            # rather than a subtler bias/precision artifact. Rendering
            # both sides is the safe choice for arbitrary/open geometry;
            # the existing depth bias in the shadow test should still
            # handle ordinary acne fine without front-face culling's help.
            self.ctx.disable(moderngl.CULL_FACE)

            for face in range(6):
                fbo = temp_shadow.live_fbos[face]
                fbo.use()
                self.ctx.viewport = (0, 0, temp_shadow.resolution, temp_shadow.resolution)
                fbo.clear(depth=1.0)
                light_vp = temp_shadow.light_mvps[face]
                for obj in self.static_objects:
                    light_mvp = light_vp * self._get_model_matrix(obj)
                    self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                    obj["shadow_vao"].render()

            for obj in eligible:
                bake_point_light(
                    self.ctx, self.bake_program, obj, self._get_model_matrix(obj),
                    light, temp_shadow
                )

            temp_shadow.destroy()

        for obj, path in zip(eligible, cache_paths):
            texture = obj["lightmap_texture"]
            width, height = texture.size
            array = np.frombuffer(texture.read(), dtype=np.float16).reshape(height, width, 4)
            np.save(path, array)

        print(f"[Scene] Baked and saved {len(eligible)} lightmap(s) to {self.lightmap_dir}")

        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.screen.use()
        self.ctx.viewport = restore_viewport
        self.ctx.depth_func = "<"
        self.ctx.cull_face = "back"

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

        if self.bake_program is not None:
            try:
                self.bake_program.release()
            except Exception:
                pass

            self.bake_program = None

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

        lightmap_vao = obj.get(
            "lightmap_vao"
        )

        if lightmap_vao is not None:
            try:
                lightmap_vao.release()
            except Exception:
                pass

        lightmap_texture = obj.get(
            "lightmap_texture"
        )

        if lightmap_texture is not None:
            try:
                lightmap_texture.release()
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