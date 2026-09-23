from __future__ import annotations

import contextlib
import time
from pathlib import Path

import moderngl
import glm
import numpy as np

from Modules.Audio.sound_manager import SoundManager
from Modules.Audio.footstep_materials import get_footstep_sound
from Modules.Physics.physics_world import PhysicsWorld, CollisionGroup, to_physics_vec
from Modules.Graphics.pbr_shader import (
    create_program,
    bind_material,
    bind_frame_uniforms,
    bind_point_lights,
    bind_environment,
    bind_reflection_environment,
    bind_ssr_textures,
    MAX_POINT_LIGHTS,
)
from Modules.Graphics.frustum import extract_frustum_planes, aabb_outside_frustum
from Modules.Graphics.shadow_module import CascadedShadowMap
from Modules.Graphics.point_shadow_module import PointShadowMap
from Modules.Graphics.gltf_lights import extract_punctual_lights
from Modules.Graphics.lightmap_baker import (
    create_bake_program,
    create_lightmap,
    bake_point_light,
    bake_directional_light,
    create_dilate_program,
    create_dilate_quad_vao,
    dilate_lightmap,
)
from Modules.Graphics.model_loader import load_glb, load_glb_by_material
from Modules.Graphics import lightmap_cache_io
from Modules.Graphics.skeletal_loader import load_skinned_glb, create_skeletal_vao, load_animation_clips
from Modules.Graphics.skeletal_shader import (
    create_skeletal_program,
    create_skeletal_shadow_program,
    bind_bone_matrices,
    upload_bone_matrices,
)
from Modules.Graphics.skybox import (
    create_skybox_program,
    load_skybox_textures,
    create_skybox_vao,
    render_skybox,
    create_equirect_skybox_program,
    load_equirect_texture,
    create_equirect_skybox_vao,
    render_equirect_skybox,
)


# Default crossfade time (seconds) between two animation clips on a
# skeletal object - see set_skeletal_animation/set_skeletal_upper_
# animation. Short enough that a state change (e.g. idle -> walk) still
# feels immediate, long enough to smooth over the pop a hard instant cut
# between two different clips would otherwise show.
_DEFAULT_ANIM_BLEND_DURATION = 0.25

# Temporary profiling aid - set True (or flip via a debugger/console) to
# time every render pass/object via real GPU timer queries (moderngl's
# ctx.query(time=True), i.e. actual GPU execution time - NOT CPU wall-
# clock time around a draw call, which would mostly measure how long it
# took to SUBMIT the call, not how long the GPU actually spent on it,
# since GL draw calls queue asynchronously) and print a sorted
# most-expensive-first breakdown every _PROFILE_RENDER_WINDOW_FRAMES
# frames - see Scene._profiled/Scene.render's own use of it. Safe to
# leave False permanently (a single boolean check per pass/object,
# nothing else runs); remove entirely once no longer needed.
PROFILE_RENDER = False
_PROFILE_RENDER_WINDOW_FRAMES = 60

# CPU wall-clock companion to PROFILE_RENDER above - that one times GPU
# EXECUTION via timer queries (see Scene._profiled's own docstring), and
# says nothing about the Python-side cost of just ISSUING each pass's
# calls, which is what app.py's own PROFILE_CPU "scene.render (cpu
# submit)" bucket actually measures as one lump sum for this whole
# method. This breaks THAT lump sum down by render()'s own sub-passes
# (shadows/main scene/skybox/ssr/transparent) using plain time.
# perf_counter, same shape as app.py's _profiled_cpu - off by default,
# a single boolean check per pass when off.
PROFILE_RENDER_CPU = False
_render_cpu_samples = {}
_render_cpu_frame_count = 0

# Temporary perf-comparison switch - set False to skip the real-time
# shadow pass entirely (_render_shadows never runs, and every
# bind_frame_uniforms call gets shadow_manager forced to None, so the
# fragment shader's own u_has_shadows reads as 0 too, which also
# zeroes calculate_movable_shadow's own result - see that function's
# own comment on why it reads the same u_has_shadows/u_shadow_maps as
# calculate_shadow now). Baked static lighting is unaffected either
# way - only the real-time cascades/sampling are skipped. Revert to
# True when done comparing.
ENABLE_SHADOWS = True


def _release(resource):
    """Best-effort GL resource release - swallows errors since this is
    always called during cleanup/replacement, never on a hot path where
    a failure should be visible."""
    if resource is not None:
        try:
            resource.release()
        except Exception:
            pass


def _resolve_upper_rotation_offsets(skeleton, upper_root_indices, degrees_spec):
    """Builds the {joint_index: glm.quat} map Skeleton.
    compute_blended_bone_matrices' own upper_rotation_offsets expects,
    from add_skeletal/set_skeletal_upper_rotation_offset's
    upper_rotation_offset_degrees - see that param's own docstring for
    the two accepted forms. A dict input is keyed by joint NAME (matched
    the same way resolve_joint_indices matches any other joint name -
    silently skipping one that doesn't exist on this skeleton) so each
    root can get an independent correction; a plain (x, y, z) tuple
    applies identically to every joint in upper_root_indices."""
    if isinstance(degrees_spec, dict):
        offsets = {}
        for name, degrees in degrees_spec.items():
            for index in skeleton.resolve_joint_indices([name]):
                offsets[index] = glm.quat(glm.radians(glm.vec3(degrees)))
        return offsets
    offset_quat = glm.quat(glm.radians(glm.vec3(degrees_spec)))
    return {i: offset_quat for i in upper_root_indices}


def _apply_upper_rotation_offsets(obj, new_offsets, blend_duration=_DEFAULT_ANIM_BLEND_DURATION):
    """Replaces obj["upper_rotation_offsets"] with new_offsets, starting
    a crossfade FROM whatever was active a moment ago instead of
    snapping straight to the new correction - shared by set_skeletal_
    upper_rotation_offset (a direct correction change) and
    set_skeletal_upper_joint_mask (which resets offsets to {} as part of
    switching masks - see its own docstring) so both go through the same
    blend rather than one of them being an instant cut. See Scene.
    update()'s own per-frame blend (mirrors anim_blend_elapsed/duration's
    existing pattern for clip crossfades) for how upper_rotation_offsets_
    prev/upper_offset_blend_elapsed actually get consumed - a joint
    present in one dict but not the other fades from/to the identity
    quaternion (no correction), which is exactly right for e.g. widening
    the joint mask from clavicles to Spine4: the old clavicle corrections
    ease OUT to nothing while Spine4's new one eases IN, rather than
    either popping."""
    obj["upper_rotation_offsets_prev"] = obj["upper_rotation_offsets"]
    obj["upper_rotation_offsets"] = new_offsets
    obj["upper_offset_blend_elapsed"] = 0.0
    obj["upper_offset_blend_duration"] = blend_duration


def _advance_clip_time(skeleton, obj, dt, anim_key="animation", time_key="anim_time", loop_key="anim_loop"):
    """Advances obj[time_key] by dt against obj[anim_key]'s own clip
    duration - the single-clip time-advance every non-blend-space track
    needs, whether it's the lower body, the upper body, or an object with
    no split at all: looping via modulo, or clamping at the clip's own
    duration to hold its last frame if loop_key reads False (see
    set_skeletal_animation's own loop param for why - e.g. a one-shot
    jump takeoff pose that should freeze mid-air). A no-op (resets
    time_key to 0.0) if obj[anim_key] is None or not a real/zero-length
    clip on this skeleton."""
    clip = skeleton.animations.get(obj[anim_key])
    if clip is not None and clip.duration > 0.0:
        obj[time_key] = (
            (obj[time_key] + dt) % clip.duration if obj[loop_key]
            else min(obj[time_key] + dt, clip.duration)
        )
    else:
        obj[time_key] = 0.0


def _advance_upper_offset_blend(obj, dt):
    """Advances upper_offset_blend_elapsed by dt and returns
    (blended_offsets, offset_blend_weight) - the crossfaded upper_
    rotation_offsets to use this frame, and the same weight Scene.
    update()'s mask-transition blend also uses (see _apply_upper_
    rotation_offsets and compute_blended_bone_matrices's own mask_blend_
    weight docstring for why a joint present in one dict but not the
    other fades to/from the identity quaternion). Shared by both the
    single-clip and locomotion-blend-space skeletal update paths in
    Scene.update(), which otherwise duplicate this exact computation."""
    obj["upper_offset_blend_elapsed"] = min(obj["upper_offset_blend_elapsed"] + dt, obj["upper_offset_blend_duration"])
    offset_blend_weight = (
        1.0 if obj["upper_offset_blend_duration"] <= 0.0
        else obj["upper_offset_blend_elapsed"] / obj["upper_offset_blend_duration"]
    )
    prev_offsets = obj["upper_rotation_offsets_prev"]
    new_offsets = obj["upper_rotation_offsets"]
    if offset_blend_weight >= 1.0:
        blended_offsets = new_offsets
    elif prev_offsets or new_offsets:
        identity_rotation = glm.quat(1.0, 0.0, 0.0, 0.0)
        blended_offsets = {
            i: glm.slerp(prev_offsets.get(i, identity_rotation), new_offsets.get(i, identity_rotation), offset_blend_weight)
            for i in (prev_offsets.keys() | new_offsets.keys())
        }
    else:
        blended_offsets = new_offsets
    return blended_offsets, offset_blend_weight


class Scene:
    def __init__(self, ctx, recalculate_shadows=True):
        self.ctx = ctx

        # If True, bake_static_lighting() recomputes lighting from scratch
        # and saves it to disk. If False, it loads the previously-saved
        # lightmaps instead of re-baking, which is much faster - useful
        # once you're happy with the lighting and don't want to pay the
        # bake cost on every launch. Falls back to baking automatically if
        # the cached files aren't there yet (e.g. first run).
        self.recalculate_shadows = recalculate_shadows
        self.lightmap_dir = Path("Assets/Lightmaps") / self.__class__.__name__

        self.static_objects = []
        self.dynamic_objects = []
        self.skeletal_objects = []
        self.point_lights = []
        self.sound_manager = SoundManager()
        self.physics = PhysicsWorld()

        # See PROFILE_RENDER/Scene._profiled - accumulated GPU
        # nanoseconds per label across the current profiling window,
        # and how many frames have gone by since the last window was
        # printed/reset.
        self._profile_samples = {}
        self._profile_frame_count = 0

        self.light_dir = glm.vec3(0.5, 1.0, 0.8)

        # The directional (sun) light's own color and brightness -
        # separate from light_dir, which only ever controlled its
        # direction (its vector's magnitude does nothing; the shader
        # normalizes it). Defaults match the shader's old hardcoded
        # behavior exactly (implicitly white at a fixed 2.0 multiplier)
        # so existing scenes look unchanged unless a scene overrides
        # these, same pattern as light_dir itself.
        self.light_color = glm.vec3(1.0, 1.0, 1.0)
        self.light_intensity = 2.0

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
                in vec2 in_uv;

                out vec2 v_uv;

                void main()
                {
                    v_uv = in_uv;
                    gl_Position = u_light_mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                // Alpha-tested shadow casting: a MASK/BLEND caster
                // (a leaf card, a chain-link fence, ...) used to cast a
                // solid shadow shaped like its whole mesh silhouette,
                // since this pass never looked at the material's own
                // alpha at all - confirmed as exactly why a cutout tree
                // canopy baked/cast a big rectangular block shadow
                // instead of a leaf-shaped one. Mirrors the real-time
                // color shader's own OPAQUE-never-discards, MASK/BLEND-
                // discard-below-cutoff logic (pbr_shader.py's main()) -
                // same u_alpha_cutoff a material already carries, no
                // new authoring needed. Applied to BLEND too, not just
                // MASK: a shadow is a binary depth write, there's no
                // such thing as a "50% transparent" shadow to write, so
                // BLEND reuses the same cutoff as the least-bad
                // approximation (the same practical shortcut real-time
                // engines take for "alpha tested shadows" on
                // translucent-looking foliage) rather than either
                // casting a fully solid shadow or none at all.
                //
                // Every shadow-CASTING call site (the real-time
                // cascades in Scene._render_shadows, and both bake-time
                // passes - Scene._bake_directional_shadow_map and
                // bake_static_lighting's own per-point-light shadow
                // cube) shares this one program, so this fix applies to
                // all of them uniformly rather than needing to be
                // duplicated per pass. u_has_texture/u_alpha_mode
                // default to 0 (moderngl zero-initializes uniforms) for
                // any call site that doesn't explicitly bind them,
                // which safely means "never discard" - exactly the old
                // behavior - rather than silently breaking whichever
                // pass hasn't been updated to bind them yet.
                uniform sampler2D u_texture;
                uniform int u_has_texture;
                uniform int u_alpha_mode;
                uniform float u_alpha_cutoff;
                uniform float u_base_alpha;

                in vec2 v_uv;

                void main()
                {
                    if (u_alpha_mode != 0 && u_has_texture == 1) {
                        float alpha = texture(u_texture, v_uv).a * u_base_alpha;
                        if (alpha < u_alpha_cutoff) {
                            discard;
                        }
                    }
                }
            """
        )

        # This cascade set holds ONLY dynamic/skeletal (movable) casters
        # - static geometry's own shadow contribution comes from self.
        # _static_shadow_texture instead (see _ensure_static_shadow_map/
        # _render_shadows), a single fixed map built once rather than
        # redrawn into a cascade every frame. Cheap to render every
        # frame: only however many dynamic/skeletal objects the scene
        # actually has - a handful of low-poly meshes, nothing like the
        # full static map.
        #
        # Passed once to bind_frame_uniforms' own shadow_manager param at
        # every call site - the fragment shader's calculate_shadow AND
        # calculate_movable_shadow both read straight off the resulting
        # u_shadow_maps/u_light_mvps (see calculate_cascade_shadow's own
        # comment), so there's only ever one binding of this cascade's
        # textures/matrices per bind_frame_uniforms call now. A second,
        # separately-fit "movable_shadow_manager" cascade set (and a
        # second GLSL uniform set/binding function to match) used to
        # exist here, rendering and binding this exact same content a
        # second time for no actual difference - confirmed as pure
        # redundant GPU work, and disproportionately expensive for
        # _render_transparent_objects specifically (a pass with few
        # objects, where this per-call fixed cost dominated far more
        # than it did in the main scene loop's many-objects case) - once
        # static casters were moved out of the main cascade set and into
        # the fixed static map instead (see git history if the old two-
        # manager split is ever needed again for some new reason).
        self.shadow_manager = CascadedShadowMap(self.ctx)
        # Built lazily, once, the first time _render_shadows runs (see
        # _ensure_static_shadow_map) - not here, since a Scene
        # subclass's own static objects haven't been added yet at this
        # point in __init__.
        self._static_shadow_texture = None
        self._static_shadow_light_vp = None
        # Deliberately higher than shadow_manager's own 2048
        # (CascadedShadowMap's default) - this single fixed map covers
        # the WHOLE static scene at once rather than
        # being tightly re-fit to just the camera's near cascade slice
        # the way the old per-frame static redraw was, so it needs more
        # texels to make up for that lost precision.
        self.static_shadow_resolution = 4096
        self.bake_program = create_bake_program(self.ctx)
        # See lightmap_baker.py's own dilate_lightmap - fills the empty
        # padding gap generate_lightmap_uvs leaves around each chart
        # with a plausible extension of that chart's own nearby baked
        # color, so bilinear sampling at a chart's own edge doesn't
        # blend toward the gap's cleared-black default (visible as a
        # dark seam/border around every lightmapped surface). Created
        # once here, same as self.bake_program - the geometry (a plain
        # fullscreen quad) and compiled program never change between
        # Scene.bake_static_lighting calls or objects.
        self.dilate_program = create_dilate_program(self.ctx)
        self.dilate_quad_vao, self.dilate_quad_vbo = create_dilate_quad_vao(self.ctx, self.dilate_program)
        self.skeletal_program = create_skeletal_program(self.ctx)
        self.skeletal_shadow_program = create_skeletal_shadow_program(self.ctx)

        self.skybox_program = create_skybox_program(self.ctx)
        self.skybox_textures = None
        self.skybox_average_colors = None
        self.skybox_vao = None
        self.skybox_vbo = None
        self.skybox_edge_fade = 0.05

        self.equirect_skybox_program = create_equirect_skybox_program(self.ctx)
        self.equirect_skybox_texture = None
        self.equirect_skybox_vao = None
        self.equirect_skybox_vbo = None
        self.equirect_exposure = 1.0
        self.equirect_is_hdr = True

        # Hemisphere ("skylight"-style) ambient term derived from
        # whichever skybox is loaded (see add_skybox/add_equirect_skybox
        # and pbr_shader.py's u_sky_color/u_ground_color) - dim neutral
        # gray until a skybox actually sets these, close to the flat
        # ambient constant this replaced.
        self.environment_sky_color = (0.025, 0.025, 0.025)
        self.environment_ground_color = (0.025, 0.025, 0.025)
        # The RAW (pre-ambient_intensity, pre-exposure-for-equirect)
        # average colors add_skybox/add_equirect_skybox actually derived
        # from the loaded texture - kept around so set_ambient_intensity
        # below can be called at any time afterward (even repeatedly, to
        # re-tune) and recompute environment_sky_color/ground_color from
        # the real base values, rather than needing the skybox reloaded
        # from disk just to change this one multiplier.
        self._base_sky_color = self.environment_sky_color
        self._base_ground_color = self.environment_ground_color
        # 1.0 = neutral (today's existing brightness, unchanged) - see
        # set_ambient_intensity's own docstring.
        self.ambient_intensity = 1.0

        # See update()'s own comment - drives pbr_shader.py's u_time.
        self._elapsed_time = 0.0
        # Plain incrementing frame counter - drives pbr_shader.py's
        # u_frame_offset, which only exists to vary ssr_reflect's dither
        # pattern frame to frame (see that uniform's own comment).
        self._frame_count = 0

        # Set up by enable_screen_space_reflections() - None (the
        # feature's default-off state) until then. See that method's
        # own docstring.
        self._ssr_fbo = None
        self._ssr_depth_fbo = None
        self._ssr_color_texture = None
        self._ssr_depth_texture = None
        self._ssr_resolution = None

    # =============================================================
    # OBJECT LOADING
    # =============================================================

    def _load_object(self, model_path):
        model = load_glb(model_path, self.ctx, self.pbr_program)
        if model is None:
            return None

        # load_textures=False - shadow_program now declares u_texture
        # too (an alpha-tested discard - see its own fragment shader),
        # but the shadow pass always reuses the PBR pass's ALREADY-
        # loaded texture at draw time instead (see Scene._bind_shadow_
        # alpha) - see _build_mesh_data's own load_textures docstring
        # for why loading one here too would be pure duplicated waste.
        shadow_model = load_glb(model_path, self.ctx, self.shadow_program, load_textures=False)

        # Only bother with a bake-program VAO if the glb actually has a
        # second UV channel to bake into - otherwise this is wasted work.
        lightmap_model = (
            load_glb(model_path, self.ctx, self.bake_program)
            if model.get("has_lightmap_uv") else None
        )

        if shadow_model is None:
            _release(model.get("vao"))
            if lightmap_model is not None:
                _release(lightmap_model.get("vao"))
            _release(model.get("texture"))
            _release(model.get("metallic_roughness_texture"))
            _release(model.get("normal_texture"))
            return None

        return {
            "name": Path(model_path).stem,
            "vao": model["vao"],
            # Kept around beyond just building the VAO above - see
            # Scene._bake_directional_shadow_map, which reads this back
            # on the CPU to compute the static scene's own world-space
            # bounds for the sun's bake-time shadow map, without a
            # second independent load of the source file.
            "vbo": model.get("vbo"),
            "shadow_vao": shadow_model["vao"],
            "lightmap_vao": lightmap_model["vao"] if lightmap_model else None,
            "has_lightmap_uv": model.get("has_lightmap_uv", False),
            "lightmap_texture": None,
            "texture": model.get("texture"),
            "metallic_roughness_texture": model.get("metallic_roughness_texture"),
            "normal_texture": model.get("normal_texture"),
            "metallic": model.get("metallic", 0.1),
            "roughness": model.get("roughness", 0.5),
            "emissive": model.get("emissive", [0.0, 0.0, 0.0]),
            "normal_scale": model.get("normal_scale", 1.0),
            "double_sided": model.get("double_sided", False),
            "has_texture": model.get("has_texture", 0),
            "has_metallic_roughness_texture": model.get("has_metallic_roughness_texture", 0),
            "has_normal_texture": model.get("has_normal_texture", 0),
            "alpha_mode": model.get("alpha_mode", "OPAQUE"),
            "alpha_cutoff": model.get("alpha_cutoff", 0.5),
            "base_alpha": model.get("base_alpha", 1.0),
        }

    def _load_objects_by_material(self, model_path):
        """Like _load_object, but for a glb that may have more than one
        distinct material - see load_glb_by_material's own docstring for
        why this exists (load_glb/_load_object's _flatten_scene only
        ever keeps ONE material for the whole merged mesh, which is
        wrong for a real multi-material level). Returns a list of dicts,
        each shaped exactly like _load_object's own single return value,
        one per distinct material actually present in model_path (empty
        list on load failure) - add_static appends ALL of them to self.
        static_objects as independent entries sharing the same
        transform, rather than nesting them under one object, so every
        other per-object code path here (render, shadow, lightmap
        baking, release) needs no changes at all to handle a multi-
        material static object: from their perspective it's just several
        static objects, which they already know how to handle."""
        pbr_groups = load_glb_by_material(model_path, self.ctx, self.pbr_program)
        if not pbr_groups:
            return []

        # load_textures=False - see _load_object's own identical call
        # for why.
        shadow_groups = load_glb_by_material(model_path, self.ctx, self.shadow_program, load_textures=False)
        if len(shadow_groups) != len(pbr_groups):
            print(
                f"[Scene] {model_path}: pbr pass produced {len(pbr_groups)} material "
                f"group(s) but the shadow pass produced {len(shadow_groups)} - giving up "
                f"on this object rather than mismatching groups across passes."
            )
            for m in pbr_groups:
                _release(m.get("vao"))
                _release(m.get("texture"))
                _release(m.get("metallic_roughness_texture"))
                _release(m.get("normal_texture"))
            for m in shadow_groups:
                _release(m.get("vao"))
            return []

        # Only bother with bake-program VAOs if at least one group
        # actually has a second UV channel to bake into - otherwise this
        # is wasted work, same reasoning as _load_object's own.
        lightmap_groups = [None] * len(pbr_groups)
        if any(m.get("has_lightmap_uv") for m in pbr_groups):
            loaded = load_glb_by_material(model_path, self.ctx, self.bake_program)
            if len(loaded) == len(pbr_groups):
                lightmap_groups = loaded
            else:
                print(
                    f"[Scene] {model_path}: pbr pass produced {len(pbr_groups)} material "
                    f"group(s) but the bake pass produced {len(loaded)} - disabling "
                    f"lightmap baking for this object rather than mismatching groups."
                )
                for m in loaded:
                    _release(m.get("vao"))

        results = []
        for i, model in enumerate(pbr_groups):
            shadow_model = shadow_groups[i]
            lightmap_model = lightmap_groups[i]
            results.append({
                "name": Path(model_path).stem,
                "vao": model["vao"],
                # See _load_object's own identical field for why this is
                # kept - Scene._bake_directional_shadow_map reads it back.
                "vbo": model.get("vbo"),
                "shadow_vao": shadow_model["vao"],
                "lightmap_vao": lightmap_model["vao"] if lightmap_model else None,
                "has_lightmap_uv": model.get("has_lightmap_uv", False),
                "lightmap_texture": None,
                "texture": model.get("texture"),
                "metallic_roughness_texture": model.get("metallic_roughness_texture"),
                "normal_texture": model.get("normal_texture"),
                "metallic": model.get("metallic", 0.1),
                "roughness": model.get("roughness", 0.5),
                "emissive": model.get("emissive", [0.0, 0.0, 0.0]),
                "normal_scale": model.get("normal_scale", 1.0),
                "double_sided": model.get("double_sided", False),
                "has_texture": model.get("has_texture", 0),
                "has_metallic_roughness_texture": model.get("has_metallic_roughness_texture", 0),
                "has_normal_texture": model.get("has_normal_texture", 0),
                "alpha_mode": model.get("alpha_mode", "OPAQUE"),
                "alpha_cutoff": model.get("alpha_cutoff", 0.5),
                "base_alpha": model.get("base_alpha", 1.0),
                # The source glTF material's own name (or id(mat) for
                # an unnamed one - see model_loader.py's _material_key)
                # - not used anywhere in the render path itself, only
                # so add_static's alpha_mode_overrides can target one
                # specific material by name.
                "material_name": model.get("material_name"),
            })
        return results

    # =============================================================
    # STATIC OBJECTS
    # =============================================================

    def add_static(self, model_path, position=None, rotation=None, scale=None,
                    transform=None, metallic=None, roughness=None,
                    collision=False, collision_shape="mesh", collision_mask=CollisionGroup.ALL,
                    collision_exclude_local_bounds=None, physical_material=None,
                    alpha_mode_overrides=None, roughness_overrides=None,
                    specular_strength_overrides=None, water_overrides=None,
                    double_sided_overrides=None, ignore_source_double_sided=False,
                    collision_overrides=None, collision_object_overrides=None,
                    base_alpha_overrides=None):
        """collision=True registers a collider for this object in
        self.physics, so a CharacterController (or a dynamic object
        with its own collision=True) can stand/collide on it.
        collision_shape: "mesh" (default) is exact per-triangle
        collision built from model_path's own geometry - the right
        choice for level geometry (floors, walls, ramps). "box" is a
        cheaper axis-aligned-in-local-space box sized from the mesh's
        bounds - fine for simple blocking volumes. collision_mask: see
        CollisionGroup / physics_world.py's module docstring for the
        collision-filtering model - the default (ALL) collides with
        everything. collision_exclude_local_bounds: only for
        collision_shape="mesh" - see PhysicsWorld.add_static_mesh's
        exclude_local_bounds docstring; carves a region out of the mesh
        collision (e.g. one being replaced by a separate simplified
        collider added alongside this call). physical_material: a name
        like "dirt"/"concrete"/"wood" (see Modules/Audio/
        footstep_materials.py for the full set, derived from Assets/
        Audio/footsteps' filenames) - tags this object's collider so
        CharacterController's footstep sounds pick the right sample set
        while standing on it. None falls back to
        footstep_materials.DEFAULT_FOOTSTEP_MATERIAL. Only meaningful
        alongside collision=True.

        collision_overrides: optional {material_name: str_or_None} -
        per-MATERIAL collision behavior, matched by the same material
        name alpha_mode_overrides/roughness_overrides use. A string
        gives every object using that ONE material its own physical_
        material (footstep sound), independent of the blanket physical_
        material above; None instead builds NO collider AT ALL for that
        material's geometry, wherever it's used. A material NOT named
        here at all just uses the blanket physical_material like before
        this param existed. Use collision_object_overrides below
        instead when the thing you actually want to target is a
        specific OBJECT, not everything sharing one material (the more
        common case for a real level, where a handful of trim materials
        get reused across many distinct objects) - e.g. Blender scenes
        with separate "Water"/"CenterFloor" objects that happen to both
        use a shared material would need collision_object_overrides,
        not this, to target just one of them. Only meaningful alongside
        collision=True and collision_shape="mesh" (see PhysicsWorld.
        add_static_mesh_by_material, which this switches to internally
        when given - a "box" collision_shape has no meaningful per-
        material split, since it's already one single shape covering
        the whole model's bounds regardless).

        collision_object_overrides: optional {node_name: str_or_None} -
        the per-OBJECT equivalent of collision_overrides above, matched
        by each mesh's own NODE name instead (Blender's "Object name",
        carried through to the glTF node name on export - open the
        source file in Blender and check the Outliner, or a raw glTF
        dump's node list, if unsure what an object is actually called;
        it's frequently NOT the same as its material's name). A string
        gives that ONE object its own physical_material; None builds no
        collider at all for just that object, leaving every OTHER
        object - even ones sharing the exact same material - completely
        unaffected. This is almost always what you want for "make THIS
        one thing non-solid" or "THIS one floor sounds different"
        requests, since real levels rarely give a truly unique material
        to every individual object. Same collision=True/collision_
        shape="mesh" requirement as collision_overrides (see PhysicsWorld
        .add_static_mesh_by_object). If BOTH this and collision_overrides
        are given for the same add_static call, this one wins OUTRIGHT
        for the whole model - the two aren't merged triangle-by-triangle
        (that would need a third, per-(object,material)-pair grouping
        this doesn't build), so collision_overrides is simply ignored,
        with a note printed, rather than silently doing something more
        clever than it actually does. Pass one or the other, not both,
        for one add_static call.

        alpha_mode_overrides: optional {material_name: "OPAQUE" |
        "MASK" | "BLEND"} - forces how ONE specific material (matched
        by its own name in the source glTF - see model_loader.py's
        _material_key) renders here, regardless of what alphaMode it
        was actually authored/exported with. Exists specifically for
        "MASK" - Blender's glTF exporter has no direct equivalent to
        Unreal's Masked (hard-cutout, effectively-opaque) material
        blend mode; the closest matching Blender setting still exports
        as alphaMode=BLEND (real alpha blending: no depth write, drawn
        in a separate sorted pass - see Scene._render_transparent_
        objects' own docstring for why that's fundamentally unable to
        correctly resolve overlapping geometry WITHIN one object, like
        a tree's own trunk/leaves/backfaces all sharing one material).
        A material that's actually just a hard cutout (every pixel's
        alpha is either ~0 or ~1, nothing genuinely soft in between -
        true of most foliage/chain-link/grate textures) can safely be
        forced to "MASK" here instead: same visual result, but rendered
        in the ordinary opaque pass (full depth write/test, no sorting
        needed, only double-sided culling changes) since a MASK
        fragment is binary discard-or-opaque, never blended. Only
        meaningful when the material's alpha channel is ACTUALLY binary
        in practice - forcing a genuinely soft/translucent material
        (real glass, water) to MASK would just replace smooth edges
        with jagged ones, not fix anything.

        roughness_overrides: optional {material_name: float} - like
        alpha_mode_overrides above but for ONE specific material's
        roughnessFactor, rather than `roughness` above which (when
        given) blanket-overrides EVERY material this model_path contains
        - use this instead when only one material in a multi-material
        level needs correcting (a material authored with a placeholder/
        wrong roughness, or one worth tuning in code without re-
        exporting the source file) while the rest should keep whatever
        they were actually authored with. Applied AFTER the blanket
        `roughness` above, so a material named here always wins over it
        even if both are given. Matched by the same material_name as
        alpha_mode_overrides (see its own docstring for exactly what
        that name is).

        base_alpha_overrides: optional {material_name: float in [0, 1]} -
        same shape/matching as roughness_overrides above, but for the
        material's own base_alpha factor (glTF baseColorFactor's alpha
        channel - see model_loader.py's _extract_material). By itself
        this does NOTHING VISIBLE for an OPAQUE material (see pbr_
        shader.py's main() - OPAQUE always outputs full alpha regardless
        of this factor, and Scene._render_scene never even enables
        GL_BLEND for the OPAQUE pass to begin with) - real transparency
        also needs alpha_mode_overrides to switch that same material to
        "BLEND" (see its own docstring), at which point this is what
        actually controls how see-through it reads: 1.0 fully opaque
        (the default - matches every material's behavior before this
        param existed), lower values progressively more transparent,
        0.0 fully invisible. A material NOT named here at all keeps
        whatever base_alpha its own source file was authored with.

        specular_strength_overrides: optional {material_name: float} -
        same shape/matching as roughness_overrides above, but for
        u_specular_strength (pbr_shader.py's FRAGMENT_SHADER_BODY -
        Source's own $phongboost equivalent: a direct multiplier on the
        specular highlight's intensity, independent of roughness/
        metallic, which only shape the highlight's SIZE/tint, not how
        bright it is). No add_static caller sets this at all by default
        (every material silently gets pbr_shader.py's own 1.0 fallback),
        which is fine for ordinary matte/semi-glossy props, but a
        near-mirror-smooth STATIC material (roughness close to 0, e.g.
        water) needs it explicitly boosted well above 1.0 to actually
        read as shiny: this shader has no Fresnel edge-brightening term
        (real water's characteristic grazing-angle glare has no
        equivalent here at all) and uses a physically-modest F0≈0.04
        dielectric base reflectance, so the specular contribution is
        also multiplied by the scene's own light_intensity - in a scene
        authored with a deliberately dim direct light (mainmap_scene.py
        uses 0.1, relying on ambient/baked lighting for its overall
        exposure instead), the resulting highlight at strength=1.0 comes
        out close to imperceptible even though every other factor
        (roughness=0, metallic=0) is completely correct - reading as
        "fully rough" despite nothing being wrong with the material data
        itself, purely because the highlight is too dim to ever notice.

        water_overrides: optional {material_name: {param: value, ...}} -
        a Source-water-recipe knob set for ONE specific material, bundled
        together (unlike the single-value overrides above) since these
        are naturally authored as a group and Source's own water VMTs
        (water_dx90 etc.) expose them the same way. Recognized keys, all
        optional within a material's own dict:
          "pan1_speed": (u, v) - UV units/second the first normal-map
            layer scrolls at. Source's own $bumptransform equivalent.
          "pan2_speed": (u, v) - same, for a SECOND, independently-
            panned sample of the SAME normal map, blended with the
            first (see pbr_shader.py's apply_normal_map) - Source's
            water shaders do exactly this (two bump layers at different
            speeds/directions) so a repeating ripple tile doesn't read
            as an obviously-scrolling grid. Both default to (0, 0) -
            i.e. no panning/animation at all - if this key or the whole
            water_overrides entry is omitted, matching every material's
            behavior before this feature existed exactly.
          "uv_scale": float - tiles BOTH layers' UVs by this factor,
            applied BEFORE uv2_scale below (which then still means
            "relative to layer 1's own tiling", not the raw mesh UVs) -
            a value > 1.0 tiles the texture MORE times across the
            surface, so the pattern reads SMALLER/denser; < 1.0 tiles it
            FEWER times, so the pattern reads BIGGER/more spread out
            (the inverse of what "scale" might suggest at a glance,
            since this scales UV coordinates, not the visual pattern
            size directly - want bigger ripples, use a SMALLER number
            here). Defaults to 1.0 - the mesh's own authored UV density,
            unmodified, matching every material's behavior before this
            key existed.
          "uv2_scale": float - tiles the SECOND layer's UVs by this
            factor relative to uv_scale/layer 1's own tiling (see above)
            - a non-1.0, non-integer value (e.g. 2.3) further breaks up
            the repeat pattern since the two layers' tiling no longer
            lines back up on a simple cycle. Defaults to 1.0 (same
            tiling as layer 1).
          "normal_scale": float - how strongly the normal map perturbs
            the lighting normal (glTF's own normalTexture.scale -
            multiplies the sampled tangent-space normal's XY before
            renormalizing, see pbr_shader.py's apply_normal_map) -
            higher means MORE pronounced/bumpier-looking ripples, lower
            means a subtler, closer-to-flat surface. Overrides whatever
            the source glTF material itself was authored with (this
            project's own water normal map is authored at 0.3) rather
            than requiring a re-export just to try a stronger look.
            Defaults to whatever the material's own file specifies
            (see model_loader.py's _extract_material) if this key is
            left out entirely.
          "reflection_mode": "cheap" (default) or "ssr" - which source
            main()'s own env_reflection block samples for this material:
            "cheap" is the plain skybox reflection every near-mirror-
            smooth material already gets (see specular_strength_
            overrides above); "ssr" ray-marches the actual rendered
            scene (Screen-Space Reflections - see Scene.enable_screen_
            space_reflections, which must be called on this scene for
            "ssr" to do anything; it silently falls back to "cheap"
            otherwise) for a real reflection of whatever's actually
            above the water that moves correctly with the camera -
            unlike an earlier mirrored-camera approach this project
            tried first, which only ever looked right from a narrow
            range of angles and visibly swam otherwise (see pbr_
            shader.py's ssr_reflect for why SSR doesn't have that
            problem: it traces the SAME camera's own view, not a
            separate one, so there's no second projection to line up).
        Only meaningful alongside a normal_texture the source material
        already has (see model_loader.py's _extract_material) - pan
        speeds/uv2_scale have nothing to animate without one, though
        reflection_mode works on any near-mirror-smooth material
        regardless.

        double_sided_overrides: optional {material_name: bool} - forces
        whether ONE specific material back-face culls (False) or not
        (True), overriding whatever Scene._render_scene would otherwise
        have decided for it (the glTF's own doubleSided flag for an
        OPAQUE material - see model_loader.py's _build_mesh_data - or
        the unconditional double-sided rendering a MASK/BLEND material
        already gets regardless of this override, since a cutout/
        translucent material reading wrong lit from one side is the
        actual reason that rule exists at all). Exists because a
        source file's own doubleSided authoring isn't always
        deliberate: many DCC tools (Blender's glTF exporter included)
        default an ordinary material to doubleSided=true unless
        "Backface Culling" was explicitly enabled on it - a re-export
        can silently flip a whole level's worth of OPAQUE materials to
        double-sided at once with nobody having actually decided that,
        which reads as broken rendering (extra draw cost, and backface
        normals/lighting showing through where they shouldn't) far more
        often than it reads as intentional.

        ignore_source_double_sided: False (default) trusts the source
        glTF's own doubleSided flag for any OPAQUE material NOT named in
        double_sided_overrides, exactly as before this pair of params
        existed. True instead treats every OPAQUE material not
        explicitly named in double_sided_overrides as back-face culled,
        full stop, regardless of what the file itself says - the
        practical fix for the "many materials suddenly double-sided
        after a re-export, nobody meant that" case above: list ONLY the
        few materials that genuinely need double-sided rendering in
        double_sided_overrides, set this to True, and every other
        material culls normally no matter how the source file's own
        doubleSided flags happen to be set. MASK/BLEND materials are
        never affected either way - see double_sided_overrides' own
        comment for why.

        Returns a LIST of the object dicts actually created - one per
        distinct material model_path contains (see _load_objects_by_
        material), almost always length 1 for an ordinary single-
        material prop, more for a multi-material level; empty on load
        failure. No current caller here uses this return value."""
        models = self._load_objects_by_material(model_path)
        if not models:
            return []

        for model in models:
            if metallic is not None:
                model["metallic"] = metallic
            if roughness is not None:
                model["roughness"] = roughness
            if alpha_mode_overrides and model.get("material_name") in alpha_mode_overrides:
                model["alpha_mode"] = alpha_mode_overrides[model["material_name"]]
            if roughness_overrides and model.get("material_name") in roughness_overrides:
                model["roughness"] = float(roughness_overrides[model["material_name"]])
            if base_alpha_overrides and model.get("material_name") in base_alpha_overrides:
                model["base_alpha"] = float(base_alpha_overrides[model["material_name"]])
            if specular_strength_overrides and model.get("material_name") in specular_strength_overrides:
                model["specular_strength"] = float(specular_strength_overrides[model["material_name"]])
            if water_overrides and model.get("material_name") in water_overrides:
                params = water_overrides[model["material_name"]]
                if "pan1_speed" in params:
                    model["normal_pan1_speed"] = tuple(float(v) for v in params["pan1_speed"])
                if "pan2_speed" in params:
                    model["normal_pan2_speed"] = tuple(float(v) for v in params["pan2_speed"])
                if "uv_scale" in params:
                    model["normal_uv_scale"] = float(params["uv_scale"])
                if "uv2_scale" in params:
                    model["normal_uv2_scale"] = float(params["uv2_scale"])
                if "normal_scale" in params:
                    model["normal_scale"] = float(params["normal_scale"])
                if "reflection_mode" in params:
                    model["reflection_mode"] = params["reflection_mode"]
            if double_sided_overrides and model.get("material_name") in double_sided_overrides:
                model["double_sided"] = bool(double_sided_overrides[model["material_name"]])
            elif ignore_source_double_sided and model.get("alpha_mode") == "OPAQUE":
                # Only OPAQUE - a MASK/BLEND material already renders
                # double-sided unconditionally regardless of this flag
                # (see Scene._render_scene's own cull-face decision), so
                # forcing it False here would be silently meaningless for
                # those, not an actual behavior change worth applying.
                model["double_sided"] = False

            if transform is not None:
                model["transform"] = glm.mat4(transform)
            else:
                model["position"] = glm.vec3(position if position is not None else glm.vec3(0.0))
                model["rotation"] = glm.vec3(rotation if rotation is not None else glm.vec3(0.0))
                model["scale"] = glm.vec3(scale if scale is not None else glm.vec3(1.0))

            # Precomputed once, here, rather than every frame - see
            # _get_model_matrix's own comment on why this is safe ONLY
            # because static geometry is provably never moved by
            # anything else in the codebase after this point.
            model["_cached_model_matrix"] = self._get_model_matrix(model)

            self.static_objects.append(model)

        # New static geometry invalidates any cached point-light shadow
        # bakes, since they only cover static objects.
        self.mark_static_dirty()

        if collision:
            # Every group shares the exact same transform (they're
            # pieces of one glb positioned as a unit), so any one of them
            # gives _add_static_collision the right position/rotation/
            # scale to work from - model_path itself (not any group's own
            # geometry) is what actually drives the collision shape.
            self._add_static_collision(
                model_path, models[0], collision_shape, collision_mask,
                collision_exclude_local_bounds, physical_material,
                collision_overrides, collision_object_overrides,
            )

        return models

        return model

    def _add_static_collision(self, model_path, model, collision_shape, collision_mask,
                               exclude_local_bounds=None, physical_material=None,
                               collision_overrides=None, collision_object_overrides=None):
        pos, rot, scl = self._collision_transform_args(model)
        if collision_overrides and collision_object_overrides:
            print(
                f"[Scene] add_static({model_path!r}): both collision_overrides and "
                f"collision_object_overrides were given - using collision_object_overrides only "
                f"(see add_static's own docstring for why these can't be merged)."
            )
        if collision_shape == "mesh" and collision_object_overrides:
            # See add_static's own collision_object_overrides docstring -
            # a SEPARATE collider per OBJECT (glTF node) instead of one
            # blanket mesh for the whole model_path, so an individual
            # object can skip collision entirely or use its own
            # physical_material regardless of what material it shares
            # with other objects.
            self.physics.add_static_mesh_by_object(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                object_overrides=collision_object_overrides, default_material=physical_material,
            )
        elif collision_shape == "mesh" and collision_overrides:
            # See add_static's own collision_overrides docstring - same
            # idea, split by MATERIAL instead of by object.
            self.physics.add_static_mesh_by_material(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                material_overrides=collision_overrides, default_material=physical_material,
            )
        elif collision_shape == "mesh":
            self.physics.add_static_mesh(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                exclude_local_bounds=exclude_local_bounds, material=physical_material,
            )
        elif collision_shape == "box":
            if collision_overrides or collision_object_overrides:
                print(
                    f"[Scene] add_static({model_path!r}): collision_overrides/"
                    f"collision_object_overrides are ignored for collision_shape='box' (a single box "
                    f"already covers the whole model's bounds regardless of material/object - see "
                    f"add_static's own docstring)."
                )
            self.physics.add_static_box_from_bounds(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                material=physical_material,
            )
        else:
            raise ValueError(f"Unknown collision_shape {collision_shape!r} for add_static - use 'mesh' or 'box'.")

    def _collision_transform_args(self, obj):
        """Returns (position, rotation, scale) in the form PhysicsWorld's
        add_* methods expect, whether obj uses a decomposed position/
        rotation/scale or a single transform-matrix override."""
        if "transform" in obj:
            scale, rot_quat, translation = glm.vec3(), glm.quat(), glm.vec3()
            skew, persp = glm.vec3(), glm.vec4()
            glm.decompose(obj["transform"], scale, rot_quat, translation, skew, persp)
            return translation, glm.eulerAngles(rot_quat), scale
        return (
            obj.get("position", glm.vec3(0.0)),
            obj.get("rotation", glm.vec3(0.0)),
            obj.get("scale", glm.vec3(1.0)),
        )

    # =============================================================
    # DYNAMIC OBJECTS
    # =============================================================

    def add_dynamic(self, model_path, position=glm.vec3(0.0), rotation=glm.vec3(0.0),
                     scale=glm.vec3(1.0), rot_speed=0.0, transform=None,
                     metallic=None, roughness=None,
                     collision=False, collision_shape="box", mass=1.0,
                     collision_mask=CollisionGroup.ALL, gravity=True, kinematic=False,
                     physical_material=None):
        """collision=True hands this object over to self.physics as a
        rigid body (mass, in kg-equivalent units) - from then on its
        position/rotation are driven by the physics simulation every
        frame (see Scene.update). collision_shape: "box" (default) or
        "sphere" are cheap primitives sized from the mesh's own
        bounds; "mesh" is a convex hull built from the mesh's actual
        vertices (Bullet requires a convex shape for anything that
        moves, so this is as exact as a dynamic body can get - use it
        for oddly-shaped props where a box/sphere would clip visibly).

        gravity=False (only meaningful alongside collision=True,
        ignored if kinematic=True) keeps full collision response -
        other bodies can still push it and it can still push them -
        but exempts just this body from falling, via a per-body
        gravity override rather than the world's own gravity. Still a
        normal DYNAMIC body otherwise, so any collision impulse (the
        player walking into it, another prop landing on it) is free to
        knock it around - fine for a floating-but-shovable obstacle,
        wrong for something that should hold a fixed path (see
        kinematic below for that case).

        kinematic=True (also only meaningful alongside collision=True)
        makes this body immovable by any physical force or collision
        impulse whatsoever - other things still collide against it
        solidly, but nothing can ever push, knock, or otherwise budge
        IT. rot_speed keeps working normally for a kinematic object
        (driving its actual transform every frame, same as with no
        collision at all) rather than being replaced by a real angular
        velocity the way it is for a plain dynamic body - use this for
        a spinning/moving obstacle that must follow an exact path
        regardless of what bumps into it. gravity is meaningless here
        (a kinematic body is never affected by it either way).

        See CollisionGroup / physics_world.py's module docstring for
        collision_mask."""
        model = self._load_object(model_path)
        if model is None:
            return None

        if metallic is not None:
            model["metallic"] = metallic
        if roughness is not None:
            model["roughness"] = roughness

        if transform is not None:
            model["transform"] = glm.mat4(transform)
        else:
            model["position"] = glm.vec3(position)
            model["rotation"] = glm.vec3(rotation)
            model["scale"] = glm.vec3(scale)

        model["rot_speed"] = float(rot_speed)
        self.dynamic_objects.append(model)

        if collision:
            self._add_dynamic_collision(
                model_path, model, collision_shape, mass, collision_mask, gravity, kinematic, physical_material,
            )

        return model

    def _add_dynamic_collision(self, model_path, model, collision_shape, mass, collision_mask,
                                gravity=True, kinematic=False, physical_material=None):
        pos, rot, scl = self._collision_transform_args(model)
        rot_speed = model.get("rot_speed", 0.0)

        # Physics (or, for a kinematic body, THIS method's own caller -
        # see Scene.update) owns position/rotation from here on -
        # replace any transform-matrix override with the equivalent
        # decomposed fields so _get_model_matrix keeps following it
        # every frame. For a plain dynamic body, rot_speed itself is
        # zeroed here (Scene.update() ignores it entirely for anything
        # with a _physics_body that isn't _kinematic - it'd otherwise
        # fight the physics rotation applied each frame) but its VALUE
        # is kept above and re-applied below as a real angular velocity
        # on the rigid body instead, so a spinning prop keeps spinning
        # through actual physics rather than silently stopping the
        # moment collision is turned on. A kinematic body keeps
        # rot_speed working exactly as it always did (Scene.update
        # applies it directly to model["rotation"] every frame, same
        # as a non-colliding object) since IT drives the transform,
        # not the other way around.
        model.pop("transform", None)
        model["position"] = pos
        model["rotation"] = rot
        model["scale"] = scl
        if not kinematic:
            model["rot_speed"] = 0.0
        model["_kinematic"] = kinematic

        if collision_shape == "box":
            body = self.physics.add_dynamic_box_from_bounds(
                model_path, position=pos, rotation=rot, scale=scl, mass=mass,
                collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
                material=physical_material,
            )
        elif collision_shape == "sphere":
            body = self.physics.add_dynamic_sphere_from_bounds(
                model_path, position=pos, scale=scl, mass=mass,
                collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
                material=physical_material,
            )
        elif collision_shape == "mesh":
            body = self.physics.add_dynamic_mesh(
                model_path, position=pos, rotation=rot, scale=scl, mass=mass,
                collision_mask=collision_mask, gravity=gravity, kinematic=kinematic,
                material=physical_material,
            )
        else:
            raise ValueError(f"Unknown collision_shape {collision_shape!r} for add_dynamic - use 'box', 'sphere', or 'mesh'.")

        model["_physics_body"] = body

        if rot_speed != 0.0 and not kinematic:
            # Real angular velocity (render Y axis - vertical spin,
            # the same axis rot_speed always meant) rather than the
            # dead rot_speed field, so this keeps spinning through
            # actual physics and still fully participates in collision
            # (something bumping into it interacts with its real,
            # continuously-changing orientation). Not applicable to a
            # kinematic body - it ignores velocity-based integration
            # entirely by design, which is exactly what makes it
            # immovable; rot_speed drives it directly instead, pushed
            # into the physics transform each frame by Scene.update.
            body.node().setAngularVelocity(to_physics_vec(glm.vec3(0.0, rot_speed, 0.0)))

    # =============================================================
    # SKELETAL (ANIMATED) OBJECTS
    # =============================================================

    def add_skeletal(self, model_path, position=None, rotation=None, scale=None,
                      transform=None, animation=None, metallic=None, roughness=None,
                      emissive=None, texture_path=None,
                      visible_in_color=True, cast_shadow=True,
                      upper_body_root_joints=None, upper_animation=None,
                      loop=True, upper_loop=True,
                      upper_rotation_offset_degrees=(0.0, 0.0, 0.0),
                      time_scale=1.0):
        """Loads a skinned/animated glb - see skeletal_loader.py for
        format constraints (one skin, one mesh primitive, LINEAR/STEP
        interpolation only).

        The glb's own material (base color texture, metallic/roughness
        factors, emissive, metallic-roughness texture) is used
        automatically. Pass metallic/roughness/emissive/texture_path
        explicitly only if you want to override what's actually authored
        in the file.

        animation: name of the clip to start playing immediately. Pass
        None to start in the bind/rest pose with nothing playing yet -
        use set_skeletal_animation() later to start one.

        loop/upper_loop: whether `animation`/upper_animation repeat via
        modulo (the default, matching every existing caller) or instead
        clamp at the clip's own duration and hold its last frame once
        reached - e.g. a one-shot jump takeoff pose meant to freeze in
        mid-air until a landing event explicitly switches to something
        else, rather than looping the takeoff motion while airborne. See
        set_skeletal_animation/set_skeletal_upper_animation's own loop
        param to change this later on an already-playing clip.

        visible_in_color/cast_shadow: both default True (existing
        behavior, unchanged for every current caller). Set
        visible_in_color=False to still cast a shadow every frame (see
        _render_shadows) without ever being drawn in the normal color
        pass - built for a first-person player's own body model, which
        should shadow the ground but never actually be seen by its own
        camera. cast_shadow=False is the inverse (visible, no shadow).
        Bone matrices are still recomputed every frame in Scene.update()
        regardless of either flag, since a shadow-only object still
        needs correct bones for its shadow draw.

        upper_body_root_joints: optional list of joint names (see
        Skeleton.compute_joint_mask - a list, not a single pivot joint,
        since some rigs split into more than one subtree there). When
        given, `animation`/set_skeletal_animation drive every OTHER
        joint (the "lower body") while a second, independently-timed
        clip - set via upper_animation here or set_skeletal_upper_
        animation() later - drives just the masked joints, composited
        into one skeleton each frame (Skeleton.
        compute_blended_bone_matrices). Leave this None (the default)
        for the existing single-clip behavior - untouched for every
        current caller.

        upper_rotation_offset_degrees: a COMPONENT-space correction
        applied to upper_body_root_joints' ROOT joints only (not their
        descendants - see Skeleton.compute_blended_bone_matrices'/
        _world_matrices' own rotation_offsets docstrings for why just
        the roots, and why this is component- rather than bone- or
        world-space) every frame the upper body is driven by a separate
        clip - a practical knob for a pose authored on a rig whose bind
        orientation doesn't quite match this skeleton's own (the whole
        limb visibly rotates the wrong way despite the animation data
        itself being correct relative to ITS source rig) without needing
        to re-export or hand-edit the clip. "Component space" in the
        same sense Unreal's AnimGraph uses the term: fixed relative to
        this skeleton's own root, so e.g. "-45 on Y" reliably means the
        same turn relative to the character's own body regardless of
        whatever orientation the joint's parent bone leaves its local
        axes in (a bone-space value would only mean that by coincidence)
        - AND the correction rotates rigidly WITH the character as the
        player turns, unlike true world/level space, which would visibly
        fight that turn instead (confirmed the hard way - an earlier
        version of this used true world space and the correction
        un-rotated itself relative to the body every time the player
        turned). Two forms: a single (x, y, z) Euler tuple in degrees, applied
        identically to every root joint - or a dict
        {joint_name: (x, y, z)} for an independent correction per root,
        e.g. {"...R_Clavicle": (0,-90,0), "...L_Clavicle": (0,90,0)} -
        since a mirrored-pose bug commonly affects one side differently
        than the other (or only one side at all), a single shared value
        can't always fix both. (0,0,0) (the default) applies no
        correction at all - existing behavior, unchanged. Ignored
        entirely when upper_body_root_joints is None. See
        set_skeletal_upper_rotation_offset to change this later on an
        already-added object.

        Skeletal objects are always real-time only, never lightmap-baked
        - they're inherently dynamic (animated), so
        bake_static_lighting() never looks at this list at all, same as
        dynamic_objects already doesn't.

        time_scale: see skeletal_loader.load_skinned_glb's own
        docstring - an opt-in per-keyframe-time correction multiplier
        for a model file known to have been baked at the wrong frame
        rate. 1.0 (no change) by default."""
        data = load_skinned_glb(model_path, ctx=self.ctx, time_scale=time_scale)
        if data is None:
            return None

        render_vao_info = create_skeletal_vao(self.ctx, self.skeletal_program, data)
        shadow_vao_info = create_skeletal_vao(self.ctx, self.skeletal_shadow_program, data)

        texture = data.get("texture")
        if texture_path is not None:
            # Explicit override - release whatever the glb's own material
            # provided (if anything) and use this instead.
            _release(texture)
            from PIL import Image
            img = Image.open(texture_path).convert("RGB")
            texture = self.ctx.texture(img.size, 3, img.tobytes())
            texture.build_mipmaps()
            # Set once here - see pbr_shader.py's _bind_material_textures
            # for why this is no longer set redundantly on every frame's
            # bind_material() call instead.
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = True

        mr_texture = data.get("metallic_roughness_texture")
        final_metallic = metallic if metallic is not None else data.get("metallic", 0.1)
        final_roughness = roughness if roughness is not None else data.get("roughness", 0.5)
        final_emissive = emissive if emissive is not None else data.get("emissive", [0.0, 0.0, 0.0])

        skeleton = data["skeleton"]
        upper_joint_mask = (
            skeleton.compute_joint_mask(upper_body_root_joints)
            if upper_body_root_joints is not None else None
        )
        upper_root_indices = (
            skeleton.resolve_joint_indices(upper_body_root_joints)
            if upper_body_root_joints is not None else ()
        )
        upper_rotation_offsets = _resolve_upper_rotation_offsets(
            skeleton, upper_root_indices, upper_rotation_offset_degrees
        )

        if upper_joint_mask is not None:
            initial_bones = skeleton.compute_blended_bone_matrices(
                animation, 0.0, upper_animation, 0.0, upper_joint_mask,
                upper_rotation_offsets=upper_rotation_offsets,
            )
        elif animation is not None:
            initial_bones = skeleton.compute_bone_matrices(animation, 0.0)
        else:
            initial_bones = [glm.mat4(1.0) for _ in skeleton.joints]

        obj = {
            "name": Path(model_path).stem,
            "vao": render_vao_info["vao"],
            "shadow_vao": shadow_vao_info["vao"],
            "_render_vbos": render_vao_info["vbos"],
            "_render_ibo": render_vao_info["ibo"],
            "_shadow_vbos": shadow_vao_info["vbos"],
            "_shadow_ibo": shadow_vao_info["ibo"],
            "skeleton": skeleton,
            "animation": animation,
            "anim_time": 0.0,
            "anim_loop": bool(loop),
            # Crossfade state (see set_skeletal_animation) - "done"
            "prev_animation": None,
            "prev_anim_time": 0.0,
            "anim_blend_elapsed": _DEFAULT_ANIM_BLEND_DURATION,
            "anim_blend_duration": _DEFAULT_ANIM_BLEND_DURATION,
            # Locomotion blend-space state (see set_skeletal_locomotion) -
            # None means "not in a blend-space state", i.e. behave exactly
            # like every object before this feature existed, driven
            # purely by the single-clip animation/anim_time fields above.
            # When set, these OVERRIDE animation/anim_time for sampling
            # (see Scene.update()'s skeletal loop) - animation/upper_
            # animation are still kept up to date as a bookkeeping "what's
            # currently dominant" name so leaving the blend space (back to
            # a single clip, e.g. jump/crouch) has a real clip to crossfade
            # FROM via the ordinary prev_animation mechanism above.
            "locomotion_weights": None,
            "upper_locomotion_weights": None,
            # Normalized 0..1 gait phase, SHARED across every candidate
            # clip in locomotion_weights this frame (see Scene.update()'s
            # own skeletal loop) - each candidate is sampled at
            # locomotion_phase * that clip's OWN duration, not a shared
            # raw elapsed-seconds clock, specifically so simultaneously-
            # blended clips of DIFFERENT authored lengths (confirmed a
            # real case: this project's own directional walk clips range
            # 0.7-1.08s) stay at the SAME relative point in their stride
            # instead of drifting apart - two walk cycles blended at
            # unrelated phases (e.g. one mid-swing, one at heel-strike)
            # produces an incoherent pose (legs crossing, both forward at
            # once), which read as the animation stuttering/glitching
            # whenever several directional clips got blended in quick
            # succession (rapidly changing move_direction, e.g. rubbing
            # against a wall). Advances each frame by dt divided by
            # whichever clip is currently DOMINANT (highest-weighted)
            # own duration - see Scene.update()'s own comment - so the
            # phase's own advance RATE can wobble slightly as dominance
            # shifts between differently-timed clips, but the phase VALUE
            # itself (and therefore every blended clip's relative stride
            # position) never desyncs.
            "locomotion_phase": 0.0,
            "upper_locomotion_phase": 0.0,
            "upper_joint_mask": upper_joint_mask,
            # "Already done" (matching current, no actual mask change to
            # blend from) - see set_skeletal_upper_joint_mask/Scene.
            # update()'s own mask-transition blend, which reuses this
            # SAME upper_offset_blend_elapsed/duration timer as
            # upper_rotation_offsets below (both are set together by
            # set_skeletal_upper_joint_mask, so one shared timer keeps
            # them synchronized).
            "upper_joint_mask_prev": upper_joint_mask,
            "upper_root_indices": upper_root_indices,
            "upper_rotation_offsets": upper_rotation_offsets,
            # Crossfade state for upper_rotation_offsets itself (see
            # _apply_upper_rotation_offsets) - separate from the clip
            # crossfade above (anim_blend_elapsed/duration), since a
            # rotation-offset change and a clip change don't always
            # happen together (set_upper_rotation_offset can be called
            # on its own). "Already done" (matching prev == current, no
            # actual initial correction change to blend from).
            "upper_rotation_offsets_prev": upper_rotation_offsets,
            "upper_offset_blend_elapsed": _DEFAULT_ANIM_BLEND_DURATION,
            "upper_offset_blend_duration": _DEFAULT_ANIM_BLEND_DURATION,
            "upper_animation": upper_animation,
            "upper_anim_time": 0.0,
            "upper_anim_loop": bool(upper_loop),
            "upper_prev_animation": None,
            "upper_prev_anim_time": 0.0,
            "upper_anim_blend_elapsed": _DEFAULT_ANIM_BLEND_DURATION,
            "upper_anim_blend_duration": _DEFAULT_ANIM_BLEND_DURATION,
            "bone_matrices": initial_bones,
            "bone_ubo": None,
            "texture": texture,
            "metallic_roughness_texture": mr_texture,
            "metallic": float(final_metallic),
            "roughness": float(final_roughness),
            "emissive": list(final_emissive),
            "has_texture": 1 if texture else 0,
            "has_metallic_roughness_texture": 1 if mr_texture else 0,
            "has_lightmap_uv": False,
            "lightmap_texture": None,
            "visible_in_color": bool(visible_in_color),
            "cast_shadow": bool(cast_shadow),
        }

        if transform is not None:
            obj["transform"] = glm.mat4(transform)
        else:
            obj["position"] = glm.vec3(position) if position is not None else glm.vec3(0.0)
            obj["rotation"] = glm.vec3(rotation) if rotation is not None else glm.vec3(0.0)
            obj["scale"] = glm.vec3(scale) if scale is not None else glm.vec3(1.0)

        upload_bone_matrices(self.ctx, obj)
        self.skeletal_objects.append(obj)
        return obj

    def set_skeletal_animation(self, obj, animation_name, blend_duration=_DEFAULT_ANIM_BLEND_DURATION,
                                loop=True, start_time=0.0):
        """Switches obj to a different animation clip, restarting from
        start_time (0.0, i.e. the clip's own beginning, unless a caller
        overrides it - see start_time below) - crossfading from whatever
        pose was showing the instant this is called (frozen there, not
        still advancing) over blend_duration seconds, rather than an
        instant hard cut (see Skeleton._sample_track/_local_matrix for
        how that blend is actually computed - decomposed TRS lerp/slerp
        per joint, applied by Scene.update()'s own per-frame bone
        recompute). Pass blend_duration=0.0 for the old instant-cut
        behavior. animation_name must exist in obj["skeleton"].
        animations (print obj["skeleton"].animations.keys() to see what
        a loaded glb actually has).

        start_time: seconds into animation_name to start playback from,
        instead of the clip's own beginning - for a caller that wants
        the NEW clip to pick up already in-stride rather than resetting
        to frame 0 (e.g. a held pose resuming after some other clip was
        playing). PlayerModel's own locomotion no longer needs this for
        its directional walk clips specifically - see set_skeletal_
        locomotion instead, which drives several clips continuously via
        Skeleton._sample_weighted rather than ever hard-switching between
        them. 0.0 (the default) matches every caller before this param
        existed. Not clamped to the clip's own duration here - Scene.
        update()'s own per-frame modulo (looping) or clamp (loop=False)
        already handles any value, including one larger than the new
        clip's duration.

        loop: True (default) repeats the clip via modulo once Scene.
        update() reaches its duration, matching every caller before this
        param existed. False instead clamps anim_time at the clip's own
        duration and holds its last frame there - e.g. a one-shot jump
        takeoff pose that should freeze mid-air rather than loop the
        takeoff motion while still airborne, until something explicitly
        switches away from it (typically on a landing event).

        Also EXITS the locomotion blend-space state if obj was in one
        (see set_skeletal_locomotion) - obj["locomotion_weights"] is
        cleared to None so Scene.update()'s skeletal loop goes back to
        sampling this single clip instead of continuing to blend
        whatever weighted list was active a moment ago (that clearing is
        why the no-op guard below also checks locomotion_weights, not
        just animation_name - otherwise leaving the blend space for a
        clip that happens to already match obj["animation"]'s own
        bookkept "currently dominant" name - see Scene.update()'s own
        comment on that field - would wrongly skip the exit).

        A no-op if animation_name is already what's playing/being
        blended toward AND obj wasn't in the locomotion blend space
        (loop/start_time are NOT updated in that case either - if you
        need to change loop on an already-current clip, switch away and
        back, or call this only on the transition edge as PlayerModel's
        own state machine already does) - calling this every frame while
        a state persists (as PlayerModel's own transition-only guard
        already avoids, but a future caller might not) would otherwise
        restart the crossfade from scratch each time and it would never
        finish."""
        if animation_name == obj["animation"] and obj["locomotion_weights"] is None:
            return
        obj["prev_animation"] = obj["animation"]
        obj["prev_anim_time"] = obj["anim_time"]
        obj["anim_blend_elapsed"] = 0.0
        obj["anim_blend_duration"] = blend_duration
        obj["animation"] = animation_name
        obj["anim_time"] = float(start_time)
        obj["anim_loop"] = bool(loop)
        obj["locomotion_weights"] = None

    def set_skeletal_upper_animation(self, obj, animation_name, blend_duration=_DEFAULT_ANIM_BLEND_DURATION,
                                      loop=True, start_time=0.0):
        """The upper-body equivalent of set_skeletal_animation - same
        crossfade, loop/hold-last-frame, and start_time behavior,
        independent of the lower-body one. Only meaningful for an obj
        created with upper_body_root_joints set (see add_skeletal);
        switches just the masked joints' clip. Calling this on an obj
        without a mask configured is harmless (the fields get set but
        nothing ever reads them, since Scene.update()'s bone recompute
        only takes the blended path when obj["upper_joint_mask"] is not
        None). Also EXITS the upper-body locomotion blend space if obj
        was in one, clearing obj["upper_locomotion_weights"] to None -
        see set_skeletal_animation's own docstring for exactly why (same
        mechanism, upper-body side)."""
        if animation_name == obj["upper_animation"] and obj["upper_locomotion_weights"] is None:
            return
        obj["upper_prev_animation"] = obj["upper_animation"]
        obj["upper_prev_anim_time"] = obj["upper_anim_time"]
        obj["upper_anim_blend_elapsed"] = 0.0
        obj["upper_anim_blend_duration"] = blend_duration
        obj["upper_animation"] = animation_name
        obj["upper_locomotion_weights"] = None
        obj["upper_anim_time"] = float(start_time)
        obj["upper_anim_loop"] = bool(loop)

    def set_skeletal_locomotion(self, obj, weighted_clips, upper_weighted_clips=None,
                                 blend_duration=_DEFAULT_ANIM_BLEND_DURATION):
        """ENTERS the locomotion blend-space state - see add_skeletal's
        own locomotion_weights/upper_locomotion_weights/locomotion_phase
        docstring. weighted_clips/upper_weighted_clips: list of
        (clip_name, weight) pairs (weights need not already sum to 1 -
        see Skeleton._sample_weighted, which renormalizes); upper_
        weighted_clips=None mirrors the lower track for every joint,
        matching set_skeletal_upper_animation's own animation_name=None
        convention.

        Unlike a plain per-frame reweight (which a caller does directly -
        just assign obj["locomotion_weights"]/obj["upper_locomotion_
        weights"] to the new list every frame the blend space is already
        active, no method call needed, exactly the continuous re-weighting
        a blend space exists for), THIS call is for the discrete edge of
        actually ENTERING the blend space (e.g. landing from a jump, or
        standing up from a crouch) - it crossfades FROM whatever single
        clip was playing a moment ago (obj["animation"]/obj["upper_
        animation"], frozen via the exact same prev_animation/anim_blend_
        elapsed mechanism set_skeletal_animation already uses) over
        blend_duration seconds, and resets locomotion_phase/upper_
        locomotion_phase to 0 so the newly-entered blend space's clips
        all start from their own frame 0 rather than picking up wherever
        a previous locomotion stint left off.

        A no-op if weighted_clips/upper_weighted_clips already exactly
        match what's set (mirrors set_skeletal_animation's own no-op
        guard) - calling this every frame while already in locomotion, as
        a per-frame reweight would, must go through the direct-assignment
        path above instead, or the crossfade would restart from scratch
        every single frame and never finish."""
        if weighted_clips == obj["locomotion_weights"] and upper_weighted_clips == obj["upper_locomotion_weights"]:
            return
        obj["prev_animation"] = obj["animation"]
        obj["prev_anim_time"] = obj["anim_time"]
        obj["anim_blend_elapsed"] = 0.0
        obj["anim_blend_duration"] = blend_duration
        obj["locomotion_weights"] = weighted_clips

        obj["upper_prev_animation"] = obj["upper_animation"]
        obj["upper_prev_anim_time"] = obj["upper_anim_time"]
        obj["upper_anim_blend_elapsed"] = 0.0
        obj["upper_anim_blend_duration"] = blend_duration
        obj["upper_locomotion_weights"] = upper_weighted_clips

        obj["locomotion_phase"] = 0.0
        obj["upper_locomotion_phase"] = 0.0

    def set_skeletal_upper_rotation_offset(self, obj, degrees, blend_duration=_DEFAULT_ANIM_BLEND_DURATION):
        """Changes an already-added obj's upper_rotation_offset_degrees
        (see add_skeletal's own docstring for the two accepted forms - a
        single (x,y,z) tuple applied to every root, or a dict
        {joint_name: (x,y,z)} for independent per-side correction) at
        runtime - e.g. dialing in the right correction interactively
        rather than guessing a constant up front. (0,0,0) removes the
        correction entirely.

        Crossfades from whatever correction was active a moment ago over
        blend_duration seconds (see _apply_upper_rotation_offsets and
        Scene.update()'s own per-frame blend) rather than snapping
        straight to the new one - matching set_skeletal_animation's own
        crossfade so a pose switch that also changes the rotation offset
        (e.g. PlayerModel's pistol_idle override) eases both the
        underlying clip AND its correction in together, instead of the
        clip blending smoothly while the correction pops in instantly on
        top of it. Harmless no-op if obj wasn't created with
        upper_body_root_joints set (upper_root_indices is then empty, so
        there's nothing for the offset to apply to)."""
        new_offsets = _resolve_upper_rotation_offsets(obj["skeleton"], obj["upper_root_indices"], degrees)
        _apply_upper_rotation_offsets(obj, new_offsets, blend_duration)

    def set_skeletal_upper_joint_mask(self, obj, root_joint_names, blend_duration=_DEFAULT_ANIM_BLEND_DURATION):
        """Replaces which joints obj's upper-body clip actually drives -
        recomputes both obj["upper_joint_mask"] (via Skeleton.
        compute_joint_mask) and obj["upper_root_indices"] (via
        Skeleton.resolve_joint_indices) from root_joint_names, the same
        way add_skeletal's own upper_body_root_joints does at
        construction time. Meant for a caller that needs a WIDER (or
        just different) split for some clips than others - e.g.
        PlayerModel's set_upper_override widening the mask to
        ["...Spine4"] (which also pulls in the neck/head, since Spine4
        parents both the clavicle chains AND Neck1 in rat.glb's rig) for
        a manually-forced pose, while ordinary locomotion-driven upper-
        body poses stay rooted at just the clavicles (see [[upper-body-
        joint-split]] for why locomotion needs the narrower split: a
        gun-holding pose swapped in every frame regardless of movement
        shouldn't fight the lower body for control of the head/spine
        lean). Also clears any existing upper_rotation_offsets, fading
        them out over blend_duration (see _apply_upper_rotation_offsets)
        rather than cutting them instantly - a mask change is meant to
        replace that kind of manual per-joint correction, not stack with
        it silently held over from whatever mask was active before, but
        the OLD correction easing back to nothing while the pose itself
        is also crossfading (set_skeletal_upper_animation, typically
        called right alongside this) reads as one smooth transition
        instead of the correction snapping off mid-blend.

        The mask swap ITSELF is also blended, over the same
        blend_duration: any joint whose upper/lower assignment actually
        changes (e.g. Spine4/Neck1/Head1 when widening/narrowing between
        clavicles-only and Spine4-rooted) eases between its old and new
        pose source instead of hard-cutting the instant this is called -
        confirmed as a real, separate pop from upper_rotation_offsets'
        own crossfade (that one only smooths the CORRECTION, not the
        pose it sits on top of) - see Skeleton.compute_blended_bone_
        matrices' own upper_joint_mask_prev/mask_blend_weight docstring.

        Takes effect on the very next Scene.update()."""
        skeleton = obj["skeleton"]
        obj["upper_joint_mask_prev"] = obj["upper_joint_mask"]
        obj["upper_joint_mask"] = skeleton.compute_joint_mask(root_joint_names)
        obj["upper_root_indices"] = skeleton.resolve_joint_indices(root_joint_names)
        _apply_upper_rotation_offsets(obj, {}, blend_duration)

    def load_additional_animations(self, obj, path, rename=None, time_scale=1.0):
        """Loads animation clip(s) from a SEPARATE .glb file sharing
        obj's own armature (matched by joint NAME, not file/node order -
        see skeletal_loader.load_animation_clips) and merges them into
        obj's skeleton, so a "pose" file exported on its own (this
        project's Assets/Animations/Poses/Rifle/*.glb, sharing rat.glb's
        skeleton) can be played on this obj without re-exporting the
        mesh into every pose file. Returns the list of clip names
        actually added (after any `rename` - see load_animation_clips'
        own docstring for why that's usually needed: it's common for
        several separately-exported pose files to all name their one
        clip the same generic thing, which would otherwise collide).

        time_scale: see skeletal_loader.load_animation_clips' own
        docstring - an opt-in per-keyframe-time correction multiplier
        for a pose file known to have been baked at the wrong frame
        rate. 1.0 (no change) by default."""
        return load_animation_clips(path, obj["skeleton"], rename=rename, time_scale=time_scale)

    # =============================================================
    # SKYBOX
    # =============================================================

    def add_skybox(self, face_paths, tint=None, rotations=None, top_height=1.0,
                    bottom_height=-1.0, half_extent=1.0, padding=0, edge_fade=0.05):
        """face_paths: sequence of exactly 6 image file paths, in order
        +X, -X, +Y, -Y, +Z, -Z (same face-order convention already used
        by PointShadowMap elsewhere in this project). Each face is its
        own plain 2D texture at its native resolution/aspect - not
        forced square or matched in size, unlike an earlier version of
        this built on a hardware cubemap (see skybox.py's module
        docstring for why that was the wrong approach here).

        tint: optional (r, g, b) color correction applied to all 6 faces
        (or a list of 6, one per face).

        rotations: optional list of 6 degree values (0/90/180/270), one
        per face - for a face whose source material rotates its texture.

        top_height/bottom_height/half_extent: box shape in world units.
        Defaults make a full symmetric cube (every face, including the
        top/bottom caps, exactly 1:1) - right when all 6 textures are the
        same square size. Since the skybox's view matrix has translation
        stripped, the camera is always effectively at local origin
        (0,0,0) inside this box - bottom_height MUST stay strictly
        negative (floor below the camera), never 0.0 or positive, or
        most viewing directions won't intersect the box's geometry at
        all (see skybox.py's create_skybox_vao docstring).

        padding: pixels of edge-replication padding added to each face
        (see skybox.py's load_skybox_textures docstring - with the
        current per-face UV mapping and clamp-to-edge already active,
        this doesn't change what's visible; it won't fix the harsh seam
        between different faces either, since that's caused by zero
        blending between separate textures, not edge sampling).

        edge_fade: UV-space margin each face fades toward black over,
        softening the seam where two different face textures meet - see
        skybox.py's render_skybox docstring. Default 0.05; set to 0.0
        for hard, unfaded edges (the original behavior).

        Only one skybox per scene - calling this again replaces the
        previous one (releasing its GPU resources first)."""
        if self.skybox_textures is not None:
            for tex in self.skybox_textures:
                tex.release()
        if self.skybox_vao is not None:
            self.skybox_vao.release()
        if self.skybox_vbo is not None:
            self.skybox_vbo.release()

        self.skybox_textures, self.skybox_average_colors = load_skybox_textures(
            self.ctx, face_paths, tint=tint, padding=padding
        )
        self.skybox_vao, self.skybox_vbo = create_skybox_vao(
            self.ctx, self.skybox_program, top_height=top_height,
            bottom_height=bottom_height, half_extent=half_extent, rotations=rotations
        )
        self.skybox_edge_fade = edge_fade

        # Approximate hemisphere ambient from the +Y/-Y face averages
        # (face order is +X,-X,+Y,-Y,+Z,-Z - see load_skybox_textures)
        # - see environment_sky_color's definition in __init__. These
        # average_colors are gamma-encoded display values (they're only
        # otherwise used for on-screen edge-fade blending), not
        # gamma-decoded to linear the way add_equirect_skybox's are, so
        # this is a cruder approximation than that path - acceptable
        # for a subtle ambient term on a shader that's already not
        # claiming photometric accuracy (see pbr_shader.py's docstring).
        self._base_sky_color = self.skybox_average_colors[2]
        self._base_ground_color = self.skybox_average_colors[3]
        self._apply_ambient_intensity()

    def add_equirect_skybox(self, path, exposure=1.0):
        """Loads a single equirectangular panorama as the skybox, sampled
        directly by direction vector - no discrete faces, so none of
        add_skybox()'s seam-blending machinery (edge_fade, average-color
        neighbor blending) applies or is needed here.

        Accepts either HDR (.exr, requires the OpenEXR package) or LDR
        (.png/.jpg/etc, via PIL, no extra dependency) - auto-detected
        from the file extension. See skybox.py's load_equirect_texture
        docstring for the tonemapping difference between the two.

        exposure: brightness multiplier applied before tonemapping - the
        standard HDRI exposure control, since raw radiance values don't
        have one inherently "correct" display brightness. Still applies
        for LDR sources too (plain brightness multiplier).

        Don't call this alongside add_skybox() in the same scene - both
        would render, wastefully (whichever draws wouldn't be dangerous,
        just redundant with the other)."""
        if self.equirect_skybox_texture is not None:
            self.equirect_skybox_texture.release()
        if self.equirect_skybox_vao is not None:
            self.equirect_skybox_vao.release()
        if self.equirect_skybox_vbo is not None:
            self.equirect_skybox_vbo.release()

        self.equirect_skybox_texture, self.equirect_is_hdr, sky_color, ground_color = load_equirect_texture(
            self.ctx, path
        )
        self.equirect_skybox_vao, self.equirect_skybox_vbo = create_equirect_skybox_vao(
            self.ctx, self.equirect_skybox_program
        )
        self.equirect_exposure = exposure

        # See environment_sky_color's definition in __init__ - already
        # linear (load_equirect_texture handles LDR gamma-decoding), so
        # just apply the same exposure multiplier the skybox itself
        # renders with, for a consistent look between the visible sky
        # and its ambient contribution. exposure is folded into the
        # BASE here (unlike ambient_intensity below) because it's also
        # what render_equirect_skybox uses to draw the visible backdrop
        # itself (see Scene.render) - the two should always agree, so
        # there's no separate "un-exposed" base worth keeping around the
        # way there is for ambient_intensity, which deliberately only
        # ever affects the ambient LIGHTING term, never the visible sky.
        self._base_sky_color = tuple(c * exposure for c in sky_color)
        self._base_ground_color = tuple(c * exposure for c in ground_color)
        self._apply_ambient_intensity()

    def _apply_ambient_intensity(self):
        """Recomputes environment_sky_color/environment_ground_color -
        what pbr_shader.py's hemisphere_ambient actually samples every
        frame (see bind_environment) - from self._base_sky_color/
        _base_ground_color (whatever add_skybox/add_equirect_skybox last
        derived from the loaded texture, exposure already folded in for
        the equirect case - see that method's own comment) times self.
        ambient_intensity. Called by both skybox loaders after they set
        a new base, and by set_ambient_intensity whenever the multiplier
        itself changes - either one alone is a no-op without the other
        also having run at least once, which is exactly why this is its
        own small shared step instead of being duplicated inline in
        three places."""
        self.environment_sky_color = tuple(c * self.ambient_intensity for c in self._base_sky_color)
        self.environment_ground_color = tuple(c * self.ambient_intensity for c in self._base_ground_color)

    def set_ambient_intensity(self, multiplier):
        """Scales the skybox-derived hemisphere ambient term (pbr_
        shader.py's u_sky_color/u_ground_color - the soft, direction-
        less "skylight" every surface picks up regardless of the sun/
        point lights, blended by how much it faces up vs down - see
        FRAGMENT_SHADER_BODY's own hemisphere_ambient comment) by
        `multiplier`, WITHOUT touching how bright the skybox itself
        looks on screen - unlike add_equirect_skybox's own `exposure`,
        which affects both (see that method's own comment for why the
        two intentionally don't share one knob). 1.0 is neutral (today's
        actual brightness, unchanged); > 1.0 brightens just the ambient
        contribution (useful when a scene's own light_intensity is
        deliberately dim - e.g. MainMapScene's 0.05 - and surfaces facing
        away from the sun are reading too dark/flat as a result); < 1.0
        dims it.

        Safe to call before OR after add_skybox/add_equirect_skybox (or
        neither, or repeatedly, to re-tune) - always recomputes from
        whatever base color is currently on file (the dim default gray
        from __init__ if no skybox has been loaded yet at all), never
        needs the skybox reloaded from disk just to change this."""
        self.ambient_intensity = float(multiplier)
        self._apply_ambient_intensity()

    def enable_screen_space_reflections(self):
        """Sets up real Screen-Space Reflections (SSR) - a per-frame
        "grab" of the already-rendered opaque scene's own color+depth
        (Scene._grab_scene_textures, called automatically from render()
        once this is enabled), ray-marched by any static material with
        water_overrides "reflection_mode": "ssr" (see add_static's own
        docstring) to reflect whatever's ACTUALLY there - the "reflects
        what's above it" mode, replacing an earlier mirrored-camera
        planar-reflection approach that visibly swam/drifted as the
        camera moved (a naive screen-space UV sample only lines up
        correctly for a perfectly flat, screen-aligned mirror - it isn't
        a real reflection technique on its own). SSR instead ray-marches
        the MAIN camera's own depth buffer per reflective fragment - no
        second camera, no assumption the reflective surface is even
        flat, and it moves correctly with the camera because it IS the
        camera's own view, just traced further along a bounced ray. This
        is the standard, most common real-time approximation (used by
        Unreal/Unity/etc.) short of full ray tracing - see pbr_shader.
        py's own ssr_reflect for the actual view-space linear-march +
        thickness-test + binary-refine algorithm.

        No water_height (or any per-material plane assumption) needed
        at all, unlike the old planar approach - SSR only ever needs
        per-pixel depth, which every material contributes automatically.

        Sized to match self.ctx.screen exactly (a grab-and-blit only,
        not a full second scene render like planar reflection was - see
        _grab_scene_textures) - no meaningful sharpness/performance
        trade-off to expose here the way the old resolution param was.

        Calling this again is safe (releases the old textures first) but
        does nothing useful, since there's nothing left to reconfigure.
        There is no way to fully disable this again short of building a
        new Scene; nothing in this project currently needs that.

        _ssr_fbo (color) and _ssr_depth_fbo (depth) are deliberately TWO
        separate framebuffers, not one with both attachments - see
        _grab_scene_textures/_render_ssr_depth_prepass's own comments
        for why: depth is no longer obtained by blitting self.ctx.
        screen's own depth buffer at all (confirmed as the actual cause
        of SSR reading as the cheap skybox-only fallback on Intel
        integrated graphics - see this method's git history for the
        earlier, wrong "it's MSAA" theory this replaced once real
        diagnostic data ruled it out: self.ctx.screen.samples was 0)."""
        if self._ssr_fbo is not None:
            self._ssr_fbo.release()
        if self._ssr_depth_fbo is not None:
            self._ssr_depth_fbo.release()
        if self._ssr_color_texture is not None:
            self._ssr_color_texture.release()
        if self._ssr_depth_texture is not None:
            self._ssr_depth_texture.release()

        size = tuple(self.ctx.screen.size)
        # 4 components (RGBA) - matches self.ctx.screen's own typical
        # format (this texture's own copy_framebuffer SOURCE, see
        # _grab_scene_textures) more closely than 3 (RGB) does, a real
        # portability improvement in its own right even though it wasn't
        # the actual root cause of the Intel bug this method's own
        # docstring references (that turned out to be depth, not color -
        # see below). The fragment shader already only ever reads .rgb
        # back out of u_scene_color (see ssr_reflect), so the extra
        # alpha channel here is otherwise unused.
        self._ssr_color_texture = self.ctx.texture(size, 4)
        self._ssr_color_texture.repeat_x = self._ssr_color_texture.repeat_y = False
        self._ssr_color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # COLOR ONLY - no depth_attachment. self.ctx.screen's own depth
        # buffer is never explicitly requested at a specific format
        # (window.py's WindowManager sets no GL_DEPTH_SIZE attribute at
        # all), so its ACTUAL format is whatever the driver negotiates -
        # moderngl's own ctx.depth_texture() always creates a 32-bit
        # FLOATING-POINT depth texture (confirmed via its own .dtype
        # reading "f4"), which the window's default-framebuffer depth
        # buffer can basically never match on Windows/WGL (a floating-
        # point depth buffer for the DEFAULT framebuffer specifically
        # isn't something standard pixel-format negotiation can even
        # request - float depth is essentially an FBO-only feature in
        # practice). glBlitFramebuffer requires an EXACT depth-format
        # match; Nvidia/AMD's drivers were silently tolerating (or
        # working around) that mismatch, Intel's driver correctly
        # rejects it with GL_INVALID_OPERATION per spec - confirmed
        # directly via this scene's own temporary SSR diagnostic log.
        # Keeping this FBO color-only sidesteps the mismatch entirely:
        # there's no depth attachment left for a color blit into it to
        # ever need to touch.
        self._ssr_fbo = self.ctx.framebuffer(color_attachments=[self._ssr_color_texture])

        # Populated by a real camera-space depth-only RENDER
        # (_render_ssr_depth_prepass) every frame instead of a blit from
        # self.ctx.screen - see that method's own docstring. Still a
        # moderngl-default (float32) depth texture; the ONLY thing that
        # changes is HOW it gets filled in, not its own format, since
        # there's no longer any cross-framebuffer format-matching
        # requirement to satisfy at all once nothing is blit INTO it.
        self._ssr_depth_texture = self.ctx.depth_texture(size)
        self._ssr_depth_texture.repeat_x = self._ssr_depth_texture.repeat_y = False
        self._ssr_depth_fbo = self.ctx.framebuffer(depth_attachment=self._ssr_depth_texture)
        self._ssr_resolution = size

    # =============================================================
    # POINT LIGHTS
    # =============================================================

    def add_point_light(self, position, color=(1.0, 1.0, 1.0), intensity=1.0,
                         radius=10.0, cast_shadows=False):
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
            "bake_shadows": bool(cast_shadows),
        }
        self.point_lights.append(light)
        return light

    def add_lights_from_glb(self, model_path, cast_shadows=False, default_radius=8.0,
                             intensity_multiplier=1.0, radius_multiplier=1.0):
        """Reads KHR_lights_punctual lights out of a glb and adds any
        point lights found as real point lights in the scene.

        intensity_multiplier: scales the converted intensity (see the
        unit-conversion note in gltf_lights.extract_punctual_lights) -
        useful since the candela->linear conversion is only approximate
        and glTF-authored lights often end up too dim/bright as-is.

        radius_multiplier: scales the light's falloff radius (from the
        glTF "range" field, or default_radius if range wasn't authored).
        Useful because "range" is just a culling hint in the glTF spec,
        not a value tuned for your shader's specific falloff curve."""
        added = []

        for light in extract_punctual_lights(model_path):
            if light["type"] != "point":
                print(
                    f"[Scene] Skipping '{light['type']}' light from "
                    f"{model_path} - only 'point' lights are supported "
                    f"(no spot cone / directional handling)."
                )
                continue

            added.append(self.add_point_light(
                position=light["position"],
                color=light["color"],
                intensity=(light["intensity"] or 1.0) * intensity_multiplier,
                radius=(light["range"] or default_radius) * radius_multiplier,
                cast_shadows=cast_shadows,
            ))

        return added

    def _nearest_point_lights(self, position):
        """Returns the MAX_POINT_LIGHTS entries of self.point_lights
        closest to world-space `position`, nearest first - what a real-
        time-lit draw call (a dynamic/skeletal object, or any static
        object that somehow has no lightmap - see this method's own call
        sites in _render_scene/_render_transparent_objects) actually
        binds via bind_point_lights, instead of an arbitrary first-
        MAX_POINT_LIGHTS slice by list/insertion order (what a single
        frame-level bind_point_lights(self.pbr_program, self.
        point_lights) call - still fine for STATIC lightmapped geometry,
        which never reads these uniforms at all - would otherwise use
        for EVERY real-time-lit object regardless of where it actually
        is).

        MAX_POINT_LIGHTS (pbr_shader.py's own real-time shader array
        size, currently 4) is a hard, deliberate cap - NOT something to
        just raise to cover however many lights a level actually has
        (see pbr_shader.py's own module docstring: a larger per-light
        uniform array previously caused a real "Constant register limit
        exceeded" GLSL link error on some hardware once light counts
        grew). Scene.bake_static_lighting is what lets STATIC geometry
        see an unlimited number of lights despite this cap by baking
        them instead; a dynamic/skeletal object has no such option (its
        transform changes every frame, so nothing about its lighting can
        be pre-baked) - the best a real-time draw can do within the cap
        is make sure the few lights it DOES get are the ones that
        actually matter for wherever it currently is. Confirmed as the
        actual cause of dynamic/skeletal objects reading far less lit
        than nearby static geometry in a scene with more point lights
        than the cap (mainmap.glb has 11; without this, only the same
        first 4 in file order were EVER used for any real-time draw all
        game long, regardless of whether those 4 happened to be
        anywhere near a given dynamic object).

        No-op cost when there are MAX_POINT_LIGHTS or fewer lights in
        the whole scene to begin with (returns self.point_lights as-is,
        skipping the sort entirely) - only scenes that actually exceed
        the cap pay for this at all."""
        if len(self.point_lights) <= MAX_POINT_LIGHTS:
            return self.point_lights

        def _distance_sq(light):
            delta = light["position"] - position
            return glm.dot(delta, delta)

        return sorted(self.point_lights, key=_distance_sq)[:MAX_POINT_LIGHTS]

    def mark_static_dirty(self):
        """No-op. Kept only so existing call sites like add_static() don't
        break. There's no runtime point-light shadow cache to invalidate
        anymore - point lights are always unshadowed in real time, and
        baked shadows are recomputed by explicitly calling
        bake_static_lighting() again, not by a dirty flag."""
        pass

    # =============================================================
    # 3D SOUND
    # =============================================================

    def update_audio(self, camera):
        """Call once per frame from your main loop. Sound emitters
        themselves live on self.sound_manager - use
        scene.sound_manager.add_sound(...) to add one, not a method on
        Scene (see Modules/Audio/sound_manager.py)."""
        self.sound_manager.update(camera)

    def play_footstep_sound(self, material, position, volume=1.0):
        """One-shot footstep sample for `material` (see Modules/Audio/
        footstep_materials.py - falls back to
        footstep_materials.DEFAULT_FOOTSTEP_MATERIAL if material is
        None or unrecognized), played as a normal positional emitter
        through self.sound_manager - same distance falloff/panning
        every other 3D sound in the scene gets (see SoundManager.update),
        so footsteps attenuate with distance exactly like everything
        else rather than needing separate logic here.

        Intended to be driven by CharacterController.pop_footstep() -
        see app.py's main loop - once per footstep, not once per frame.
        loop=False means SoundManager.update automatically drops the
        emitter once playback finishes, so nothing here needs to track
        or manually destroy the sound afterward."""
        sound_path = get_footstep_sound(material)
        if sound_path is None:
            return None
        return self.sound_manager.add_sound(
            sound_path, position, volume=volume,
            min_distance=1.0, max_distance=12.0, loop=False,
        )

    # =============================================================
    # LIGHTMAP BAKING
    # =============================================================

    def _bake_directional_shadow_map(self, resolution):
        """Builds a single orthographic depth texture covering every
        static object's world-space bounds, seen from self.light_dir -
        the sun's own shadow-caster pass for bake_static_lighting,
        built ONCE per bake call rather than per object or per light:
        unlike a point light's position-and-radius-scoped shadow cube
        (see PointShadowMap), a directional light has no position for a
        per-light shadow volume to be scoped to, so every eligible
        object can share this same one map.

        Bounds are read back directly from each static object's own
        position VBO (obj["vbo"] - see model_loader.py's _build_mesh_
        data) rather than reloading the source file a second time
        (physics_world.py does that for collision, but that's a
        different, already-necessary independent load for a different
        purpose) - transformed into world space with the same model
        matrix _get_model_matrix already builds for rendering, via
        plain numpy rather than a per-vertex glm loop (verified the
        column-major byte layout .to_bytes() produces lines up with
        this row-vector-times-matrix form, matching glm's own
        model * vec4(p, 1) exactly).

        Returns (depth_texture, light_vp) - light_vp is projection *
        view (no model matrix folded in yet, same convention as
        PointShadowMap.light_mvps' own per-face entries before a
        caller multiplies in an object's model matrix). Caller owns
        depth_texture and must release it once done baking against it."""
        mins = glm.vec3(float("inf"))
        maxs = glm.vec3(float("-inf"))
        for obj in self.static_objects:
            vbo = obj.get("vbo")
            if vbo is None:
                continue
            positions = np.frombuffer(vbo.read(), dtype="f4").reshape(-1, 3).astype("f8")
            if len(positions) == 0:
                continue
            model_np = np.frombuffer(
                self._get_model_matrix(obj).to_bytes(), dtype="f4"
            ).reshape(4, 4).astype("f8")
            homogeneous = np.concatenate(
                [positions, np.ones((len(positions), 1), dtype="f8")], axis=1
            )
            world = homogeneous @ model_np
            obj_min, obj_max = world[:, :3].min(axis=0), world[:, :3].max(axis=0)
            mins = glm.min(mins, glm.vec3(float(obj_min[0]), float(obj_min[1]), float(obj_min[2])))
            maxs = glm.max(maxs, glm.vec3(float(obj_max[0]), float(obj_max[1]), float(obj_max[2])))

        if mins.x > maxs.x:
            # No static geometry with readable position data at all -
            # nothing to shadow against. Fall back to a small box around
            # the origin rather than feeding +-inf into glm.ortho below.
            mins, maxs = glm.vec3(-0.5), glm.vec3(0.5)

        center = (mins + maxs) * 0.5
        radius = glm.length(maxs - mins) * 0.5 + 0.5

        light = glm.normalize(glm.vec3(self.light_dir))
        if glm.length(light) < 1e-6:
            light = glm.vec3(0.5, 1.0, 0.8)
        world_up = (
            glm.vec3(0.0, 0.0, 1.0)
            if abs(glm.dot(light, glm.vec3(0, 1, 0))) > 0.95
            else glm.vec3(0, 1, 0)
        )

        light_view = glm.lookAt(center + light * (radius * 2.0 + 1.0), center, world_up)
        light_proj = glm.ortho(-radius, radius, -radius, radius, 0.01, radius * 4.0 + 2.0)
        light_vp = light_proj * light_view

        depth_texture = self.ctx.depth_texture((resolution, resolution))
        depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth_texture.repeat_x = depth_texture.repeat_y = False
        fbo = self.ctx.framebuffer(depth_attachment=depth_texture)
        fbo.use()
        self.ctx.viewport = (0, 0, resolution, resolution)
        fbo.clear(depth=1.0)

        # No culling - same reasoning as the point-light bake's own
        # shadow-cube pass in bake_static_lighting (a thin, single-sided
        # wall would otherwise vanish from this pass entirely on
        # whichever side gets culled, producing zero depth data and
        # letting the sun bleed straight through it).
        self.ctx.disable(moderngl.CULL_FACE)
        for obj in self.static_objects:
            light_mvp = light_vp * self._get_model_matrix(obj)
            self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
            self._bind_shadow_alpha(obj)
            obj["shadow_vao"].render()

        fbo.release()
        return depth_texture, light_vp

    def _ensure_static_shadow_map(self):
        """Builds this Scene's fixed, whole-level static-only shadow map
        ONCE, then caches it - see this method's own call site in
        _render_shadows. The camera-fit cascade set (self.shadow_manager)
        is tightly re-fit to the camera's current view frustum every
        frame (CascadedShadowMap.update), so
        a texture rendered against one of THEIR light_mvps would go
        stale/misaligned the instant the camera moves - nothing about
        that scheme can be cached across frames. Static geometry itself
        never moves though, so a separate, FIXED (camera-independent)
        ortho volume covering the whole static scene, built once, gets
        the same result the old per-frame "redraw every static object
        into every cascade" approach did, at a fraction of the ongoing
        cost - see Scene._render_shadows' own comment on dropping static
        objects from that loop.

        Reuses _bake_directional_shadow_map - the exact same fixed-
        ortho-covering-the-whole-scene logic bake_static_lighting
        already calls for the sun's OWN shadow-caster pass during
        baking - just at a HIGHER resolution (self.
        static_shadow_resolution, not whatever directional_shadow_
        resolution a bake call used) since this one has to stand in for
        a tightly-fit near cascade instead of just contributing to a
        lightmap, and kept alive afterward here instead of released
        once baking finishes.

        No-op once already built, or if there's no static geometry yet -
        e.g. called on the very first frame before this is a problem
        (see this method's own call site: it runs at the top of every
        _render_shadows call, but only ever does real work exactly
        once)."""
        if self._static_shadow_texture is not None or not self.static_objects:
            return
        self._static_shadow_texture, self._static_shadow_light_vp = (
            self._bake_directional_shadow_map(self.static_shadow_resolution)
        )

    def bake_static_lighting(self, lightmap_resolution=256, point_shadow_resolution=1024,
                              directional_shadow_resolution=2048):
        """Call this once, after adding all static objects and point
        lights, to bake shadow-tested point light contributions (from
        lights added with cast_shadows=True) AND the directional (sun)
        light's own shadowed contribution into each static object's
        lightmap. Lights are baked one at a time (additively blended) -
        see lightmap_baker.py's docstring for why that matters.

        The sun is baked using a single orthographic shadow map sized to
        cover every static object's world-space bounds (built once for
        this whole call by _bake_directional_shadow_map, not per
        object - a directional light has no position, so unlike a point
        light's per-light shadow cube there's nothing to build more than
        once here regardless of how many objects get baked against it).

        To avoid double-counting the sun (this bake AND a real-time
        directional term would otherwise both light every lightmapped
        surface), pbr_shader.py's runtime shader skips its own real-time
        sun calculation entirely for any object with a lightmap - see
        that file's main(). Static geometry still RENDERS into the
        real-time shadow cascades every frame regardless (see
        _render_shadows) - not for its own lighting, which is fully
        baked, but so it still correctly occludes the sun for a moving
        object (the player, any dynamic/skeletal object) standing
        behind or under it, and so a moving object still casts a
        real-time shadow onto static ground. Only dynamic/skeletal
        objects (which can't be baked - their transform changes every
        frame) still compute the sun's shading live.

        Point lights added with cast_shadows=False are not baked at all;
        they stay real-time-unshadowed only (see add_point_light)."""
        eligible = [
            obj for obj in self.static_objects
            if obj.get("has_lightmap_uv") and obj.get("lightmap_vao") is not None
        ]

        # This runs once, typically from a Scene subclass's __init__ -
        # i.e. before the app's main loop has necessarily set a real
        # window viewport. Capturing/restoring self.ctx.viewport like the
        # per-frame render passes do isn't safe here, since that snapshot
        # could just be moderngl's early default rather than the actual
        # window size - restore to the real framebuffer size explicitly
        # instead.
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

        # Named after the object (its model file's stem - see
        # _load_object) rather than a bare index, so cache files are
        # identifiable on disk (e.g. floorbase.exr instead of
        # lightmap_0.exr) - disambiguated with a _2, _3, ... suffix on
        # any repeat, since two static objects loaded from the same
        # model (or coincidentally sharing a stem) would otherwise
        # collide on the same cache file.
        seen_name_counts = {}
        cache_names = []
        for obj in eligible:
            name = obj["name"]
            seen_name_counts[name] = seen_name_counts.get(name, 0) + 1
            count = seen_name_counts[name]
            cache_names.append(name if count == 1 else f"{name}_{count}")
        cache_paths = [
            lightmap_cache_io.lightmap_cache_path(self.lightmap_dir, name)
            for name in cache_names
        ]

        def _load_cache():
            """Returns the loaded arrays if every cache file exists AND
            matches the resolutions currently being requested, else None.
            This check is what stops the cache from silently reusing
            stale data when lightmap_resolution or point_shadow_resolution
            change between runs."""
            loaded = []
            for path in cache_paths:
                array = lightmap_cache_io.load_lightmap_cache(
                    path, lightmap_resolution, point_shadow_resolution,
                    directional_shadow_resolution
                )
                if array is None:
                    return None
                loaded.append(array)
            return loaded

        if not self.recalculate_shadows:
            cached_arrays = _load_cache()
            if cached_arrays is not None:
                for obj, array in zip(eligible, cached_arrays):
                    _release(obj.get("lightmap_texture"))

                    # .astype/.tobytes() rather than relying on the loaded
                    # array's dtype directly - guards against a cache file
                    # ever ending up float32 (e.g. from a different numpy
                    # version) not matching the f2 (half-float) texture
                    # format below.
                    #
                    # 3 components (RGB), not 4: the RGBA requirement only
                    # applied during baking, because GL_RGB16F isn't a
                    # guaranteed-renderable framebuffer format. This
                    # texture is only ever sampled at runtime, never
                    # rendered into again, so that constraint doesn't
                    # apply here - no reason to carry a wasted alpha
                    # channel through disk storage and back.
                    array = array.astype(np.float16)
                    texture = self.ctx.texture(
                        (array.shape[1], array.shape[0]), 3, array.tobytes(), dtype="f2"
                    )
                    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    obj["lightmap_texture"] = texture

                return

            print(
                f"[Scene] recalculate_shadows is False but no valid cached "
                f"lightmaps found in {self.lightmap_dir} - baking instead."
            )

        print(f"[Scene] Baking lighting for {len(eligible)} static object(s)...")

        for obj in eligible:
            _release(obj.get("lightmap_texture"))
            obj["lightmap_texture"] = create_lightmap(self.ctx, lightmap_resolution)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE

        # Point lights flagged for baked shadows, one at a time. Each
        # gets its own temporary 6-face shadow cube (static geometry
        # only), used just for this bake, then destroyed.
        for light in self.point_lights:
            if not light.get("bake_shadows"):
                continue

            temp_shadow = PointShadowMap(
                self.ctx, resolution=point_shadow_resolution,
                near=0.05, far=max(light["radius"] * 2.0, 1.0)
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
                    self._bind_shadow_alpha(obj)
                    obj["shadow_vao"].render()

            for obj in eligible:
                bake_point_light(
                    self.ctx, self.bake_program, obj, self._get_model_matrix(obj),
                    light, temp_shadow
                )

            temp_shadow.destroy()

        # The sun, baked once against a single static-scene-covering
        # shadow map (see _bake_directional_shadow_map's own docstring
        # for why this needs building only once here, unlike the
        # per-light shadow cube the point-light loop above rebuilds for
        # every light).
        directional_depth, directional_light_vp = self._bake_directional_shadow_map(
            directional_shadow_resolution
        )
        self.ctx.viewport = restore_viewport
        for obj in eligible:
            # directional_light_vp is passed as-is (NOT multiplied by
            # this object's own model matrix) - calc_directional_shadow
            # in the bake shader transforms v_world_pos, which the
            # vertex shader already put in world space via u_model, so
            # folding u_model in a second time here would double-apply
            # it (unlike u_light_mvp in the shadow-CAST pass above,
            # which transforms raw object-local in_position and so does
            # need the model matrix folded in).
            bake_directional_light(
                self.ctx, self.bake_program, obj, self._get_model_matrix(obj),
                self.light_dir, self.light_color, self.light_intensity,
                directional_depth, directional_light_vp,
                directional_shadow_resolution,
            )
        directional_depth.release()

        # Disabled BEFORE dilation, not after (see the old position of
        # this same disable() call, just after the cache-save loop
        # below) - GL_BLEND was left enabled additive (ONE, ONE) all the
        # way from the point/directional light accumulation passes
        # above, and dilate_lightmap's own ping-pong (see its own
        # docstring) draws a fresh fullscreen quad into whichever of its
        # two textures is currently the destination WITHOUT ever
        # clearing it first between iterations - under leftover additive
        # blending, each of its 6 iterations didn't overwrite that
        # texture's previous contents (from 2 iterations ago, since it's
        # a 2-way ping-pong) but ADDED on top of them, compounding real
        # brightness growth across every already-correctly-lit interior
        # texel too, not just the padding gap dilation is actually meant
        # to fill - confirmed as the actual cause of baked lighting
        # reading significantly brighter overall after dilation was
        # added, independent of anything about the padding/margin
        # amount itself (that stays exactly as tuned - only this leaked
        # GL state was ever the bug).
        self.ctx.disable(moderngl.BLEND)

        # Dilate AFTER every light (points + sun) has been baked into
        # each object's lightmap, and BEFORE the cache-save loop below -
        # so a cached reload gets the already-dilated result too,
        # without needing to redo this on every load. See lightmap_
        # baker.py's dilate_lightmap for what/why.
        for obj in eligible:
            dilate_lightmap(self.ctx, self.dilate_program, self.dilate_quad_vao, obj["lightmap_texture"])

        for obj, path in zip(eligible, cache_paths):
            texture = obj["lightmap_texture"]
            width, height = texture.size
            array = np.frombuffer(texture.read(), dtype=np.float16).reshape(height, width, 4)
            # Drop the alpha channel before persisting - it's unused dead
            # weight here (see the load path above for why).
            lightmap_cache_io.save_lightmap_cache(
                path, array[:, :, :3], lightmap_resolution, point_shadow_resolution,
                directional_shadow_resolution
            )

        print(f"[Scene] Baked and saved {len(eligible)} lightmap(s) to {self.lightmap_dir}")

        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.screen.use()
        self.ctx.viewport = restore_viewport
        self.ctx.depth_func = "<"
        self.ctx.cull_face = "back"

    # =============================================================
    # MODEL MATRIX
    # =============================================================

    def _get_model_matrix(self, obj):
        # Only ever set by add_static (right after building this exact
        # dict, before it's appended to self.static_objects - see that
        # method's own comment) - NOT something this method sets on
        # itself for an arbitrary object, so there's no risk of handing
        # back a stale matrix for something that can actually move: a
        # dynamic object's position/rotation get overwritten every frame
        # by Scene.update's own physics-sync loop, and a skeletal
        # object's by PlayerModel.update/RemotePlayer.update_transform -
        # neither of those ever sets this key, so both always fall
        # through to a live recompute below regardless. Static geometry
        # has no such mutator anywhere in the codebase once add_static
        # returns, so precomputing this ONCE there and just returning
        # the same glm.mat4 every frame after is exactly equivalent to
        # recomputing it live - confirmed as a real, avoidable per-
        # object-per-frame cost (glm.translate/rotate x3/scale, each a
        # real Python/C++ binding call) for however many separate static
        # mesh groups a level actually has.
        cached = obj.get("_cached_model_matrix")
        if cached is not None:
            return cached

        if "transform" in obj:
            return glm.mat4(obj["transform"])

        model = glm.mat4(1.0)
        model = glm.translate(model, obj["position"])

        rotation = obj["rotation"]
        model = glm.rotate(model, rotation.x, glm.vec3(1.0, 0.0, 0.0))
        model = glm.rotate(model, rotation.y, glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, rotation.z, glm.vec3(0.0, 0.0, 1.0))

        model = glm.scale(model, obj["scale"])

        return model

    @staticmethod
    def _get_local_aabb(obj):
        """Returns (mins, maxs) - each a length-3 numpy array, in the
        object's own LOCAL/untransformed space - read from obj["vbo"]
        (the position VBO - see model_loader.py's _build_mesh_data)
        once and cached on the object itself (key "_aabb_local"), since
        the raw vertex positions never change after load. Returns None
        for an object with no "vbo" (skeletal objects don't have one -
        see add_skeletal - so they're simply never culled; there's only
        ever a handful of them, unlike static/dynamic geometry, so this
        is a deliberate scope limit rather than an oversight)."""
        cached = obj.get("_aabb_local")
        if cached is not None:
            return cached
        vbo = obj.get("vbo")
        if vbo is None:
            return None
        positions = np.frombuffer(vbo.read(), dtype="f4").reshape(-1, 3)
        if len(positions) == 0:
            return None
        aabb = (positions.min(axis=0), positions.max(axis=0))
        obj["_aabb_local"] = aabb
        return aabb

    def _get_world_aabb(self, obj, movable):
        """World-space (mins, maxs) for obj, or None if it has no local
        AABB to work from (see _get_local_aabb) - transforms the local
        AABB's 8 corners by obj's CURRENT model matrix and takes a new
        axis-aligned box around them, rather than just translating the
        local box, so this stays correct under rotation too (a rotated
        box's true world extent isn't the same as its local extent
        moved to a new position).

        movable=False (a static object - never moves after load, see
        Scene.static_objects) caches the result on the object itself
        (key "_aabb_world") and reuses it on every later call instead
        of recomputing - confirmed via CPU profiling that doing this
        numpy corner-transform fresh EVERY FRAME for EVERY object (the
        first version of this method) was itself a real, measurable
        cost once static object counts grew into the hundreds, on the
        same "numpy per-call overhead dominates at small array sizes"
        principle already hit once before in this codebase (see
        AnimationChannel.sample's own docstring in skeletal_loader.py).
        A static object's model matrix provably never changes, so nor
        can its world AABB.

        movable=True (a dynamic object) always recomputes - correct,
        and cheap enough given how few dynamic objects a scene actually
        has (nothing like the hundred-plus static objects a real level
        contains)."""
        if not movable:
            cached = obj.get("_aabb_world")
            if cached is not None:
                return cached

        local = self._get_local_aabb(obj)
        if local is None:
            return None
        local_min, local_max = local
        corners = np.array([
            [x, y, z]
            for x in (local_min[0], local_max[0])
            for y in (local_min[1], local_max[1])
            for z in (local_min[2], local_max[2])
        ], dtype="f8")

        model_np = np.frombuffer(
            self._get_model_matrix(obj).to_bytes(), dtype="f4"
        ).reshape(4, 4).astype("f8")
        homogeneous = np.concatenate([corners, np.ones((8, 1), dtype="f8")], axis=1)
        world_corners = (homogeneous @ model_np)[:, :3]
        # Plain Python float tuples, not numpy arrays - aabb_outside_
        # frustum unpacks/indexes these per plane per object, hundreds
        # of times a frame; numpy scalar overhead there was measured as
        # real (see that function's own docstring).
        aabb = (
            tuple(float(v) for v in world_corners.min(axis=0)),
            tuple(float(v) for v in world_corners.max(axis=0)),
        )

        if not movable:
            obj["_aabb_world"] = aabb
        return aabb

    def _is_visible(self, obj, frustum_planes, movable=False):
        """True unless obj's world AABB is PROVABLY entirely outside
        the given frustum (see frustum.py's own docstring) - an object
        with no AABB available (no "vbo" - see _get_local_aabb) is
        always considered visible, i.e. never culled, rather than
        risking hiding something this can't actually evaluate.

        movable: see _get_world_aabb - pass True for a dynamic object
        (its world AABB can't be cached, since it can move every
        frame); defaults False (cacheable) since most callers are
        checking static geometry."""
        aabb = self._get_world_aabb(obj, movable)
        if aabb is None:
            return True
        return not aabb_outside_frustum(aabb[0], aabb[1], frustum_planes)

    # =============================================================
    # PROFILING (see PROFILE_RENDER)
    # =============================================================

    @contextlib.contextmanager
    def _profiled(self, label):
        """Wraps a render pass or single draw call in a GPU timer query
        (see PROFILE_RENDER's own docstring for why this measures actual
        GPU execution time rather than CPU submission time) when
        PROFILE_RENDER is on - a complete no-op (the `with` block just
        runs, nothing allocated or queried) when it's off, so leaving
        every call site in place permanently costs nothing. Accumulates
        into self._profile_samples[label] (summed across every call with
        that same label this window, e.g. one call per object per frame
        adds up into that object's own running total) rather than
        overwriting, so Scene.render()'s own end-of-window report can
        show a true per-window total/average rather than just whatever
        the LAST frame happened to measure.

        label should identify WHAT this is timing specifically (e.g.
        "static:mainmap" or "skeletal:rat") - Scene.render()'s own
        per-object call sites build these from each obj's own "name"
        field (see _load_object/add_skeletal) prefixed by which pass/
        list it came from, so the eventual report can actually point at
        a specific mesh instead of just "static objects in general"."""
        if not PROFILE_RENDER:
            yield
            return
        query = self.ctx.query(time=True)
        with query:
            yield
        self._profile_samples[label] = self._profile_samples.get(label, 0) + query.elapsed

    # def _report_profile_window(self):
    #     """Called once per frame from render() - counts frames and, every
    #     _PROFILE_RENDER_WINDOW_FRAMES of them, prints every label from
    #     self._profile_samples sorted most-expensive-first (total
    #     milliseconds across the whole window, and the per-frame average -
    #     the average is usually the more directly useful number, e.g. "is
    #     THIS mesh alone costing 2ms of the 16.6ms budget for 60fps"),
    #     then resets for the next window. A no-op entirely while
    #     PROFILE_RENDER is off."""
    #     if not PROFILE_RENDER:
    #         return
    #     self._profile_frame_count += 1
    #     if self._profile_frame_count < _PROFILE_RENDER_WINDOW_FRAMES:
    #         return

    #     frames = self._profile_frame_count
    #     ranked = sorted(self._profile_samples.items(), key=lambda kv: kv[1], reverse=True)
    #     print(f"[render profile] over the last {frames} frame(s):")
    #     for label, total_ns in ranked:
    #         total_ms = total_ns / 1_000_000.0
    #         print(f"  {label:40s} total={total_ms:8.3f}ms  avg/frame={total_ms / frames:7.4f}ms")

    #     self._profile_samples = {}
    #     self._profile_frame_count = 0

    @contextlib.contextmanager
    def _profiled_cpu(self, label):
        """CPU wall-clock companion to _profiled above - see PROFILE_
        RENDER_CPU's own module-level comment for why this exists as a
        separate thing (GPU timer queries measure execution time, not
        the Python-side cost of issuing the calls in the first place).
        Same accumulate-then-report shape, module-level dict/counter
        instead of per-Scene-instance ones since app.py's own
        _profiled_cpu already established that pattern and there's no
        strong reason for this one to differ. No-op when PROFILE_RENDER_
        CPU is off."""
        if not PROFILE_RENDER_CPU:
            yield
            return
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        global _render_cpu_samples
        _render_cpu_samples[label] = _render_cpu_samples.get(label, 0.0) + elapsed

    def _report_render_cpu_profile_window(self):
        """Mirrors _report_profile_window exactly, for the CPU-side
        accumulator above instead of the GPU one - see PROFILE_RENDER_
        CPU's own comment."""
        global _render_cpu_frame_count, _render_cpu_samples
        if not PROFILE_RENDER_CPU:
            return
        _render_cpu_frame_count += 1
        if _render_cpu_frame_count < _PROFILE_RENDER_WINDOW_FRAMES:
            return

        frames = _render_cpu_frame_count
        ranked = sorted(_render_cpu_samples.items(), key=lambda kv: kv[1], reverse=True)
        print(f"[render cpu profile] over the last {frames} frame(s):")
        for label, total_s in ranked:
            total_ms = total_s * 1000.0
            print(f"  {label:20s} total={total_ms:8.3f}ms  avg/frame={total_ms / frames:7.4f}ms")

        _render_cpu_samples = {}
        _render_cpu_frame_count = 0

    # =============================================================
    # UPDATE
    # =============================================================

    def update(self, dt):
        # A plain running clock, seconds - the ONLY thing that drives a
        # water material's animated dual-layer normal pan (see pbr_
        # shader.py's u_time/MaterialBlock's own u_normal_pan1_speed/
        # u_normal_pan2_speed comments) - wrapped well before float32
        # precision would start to matter (a uniform this size loses
        # sub-millisecond precision past a few hours; wrapping resets
        # that budget long before it's ever reached, and pan_speed*time
        # modulo a period only ever discontinuously JUMPS the visible
        # scroll phase at the wrap instant if a material's own pan speed
        # doesn't evenly divide the period - 3600s is comfortably long
        # enough that this is once-an-hour at worst and imperceptible
        # for any sane pan speed).
        self._elapsed_time = (self._elapsed_time + dt) % 3600.0
        self._frame_count = (self._frame_count + 1) % 1000

        # Step collision/physics first so this frame's dynamic-object
        # sync below (and any CharacterController.get_position() calls
        # the caller makes after this) reflect where things just moved.
        self.physics.step(dt)

        for obj in self.dynamic_objects:
            physics_body = obj.get("_physics_body")
            if physics_body is not None:
                if obj.get("_kinematic"):
                    # WE drive a kinematic body's transform, not the
                    # other way around (see add_dynamic's docstring) -
                    # apply rot_speed exactly like a non-colliding
                    # object below, then push the result onto the
                    # physics node so Bullet uses it for collision.
                    rot_speed = obj.get("rot_speed", 0.0)
                    if rot_speed != 0.0 and "rotation" in obj:
                        obj["rotation"].y += rot_speed * dt
                    self.physics.set_transform(physics_body, obj["position"], obj["rotation"])
                else:
                    obj["position"], obj["rotation"] = self.physics.get_transform(physics_body)
                continue

            rot_speed = obj.get("rot_speed", 0.0)
            if rot_speed != 0.0 and "rotation" in obj:
                obj["rotation"].y += rot_speed * dt

        for obj in self.skeletal_objects:
            skeleton = obj["skeleton"]
            upper_mask = obj.get("upper_joint_mask")

            # Every track (lower, and upper if this obj has a split) is
            # fed into Skeleton's weighted-list sampler either way - a
            # single clip is just a 1-entry list (weight 1.0), which
            # Skeleton._sample_weighted takes a fast path for that's
            # identical in behavior/cost to the old single-clip-only
            # sampler - so a locomotion-blend-space track (see
            # set_skeletal_locomotion) and a plain single-clip track (a
            # held jump/crouch pose, an upper-body override, a decorative
            # prop with no locomotion at all) share one code path here
            # instead of two parallel ones that could quietly drift apart.
            lower_weights = obj.get("locomotion_weights")
            upper_weights = obj.get("upper_locomotion_weights") if upper_mask is not None else None

            if lower_weights is not None:
                # Bookkeeping only - not read by compute_bone_matrices_
                # multi/compute_blended_bone_matrices_multi themselves -
                # so that LEAVING the blend space (e.g. jumping) has a
                # real single clip name (and, via anim_time below, the
                # correct CURRENT point in it) to crossfade FROM via the
                # ordinary set_skeletal_animation/prev_animation
                # mechanism, without that mechanism needing any notion of
                # "weighted list" at all.
                dominant_name, dominant_weight = max(lower_weights, key=lambda nw: nw[1], default=(None, 0.0))
                dominant_clip = skeleton.animations.get(dominant_name) if dominant_name is not None else None

                # Advances locomotion_phase (0..1, wrapped) by dt divided
                # by the DOMINANT clip's own duration, then samples every
                # candidate clip at that SAME phase fraction of ITS OWN
                # duration - see add_skeletal's own locomotion_phase
                # docstring for why a shared free-running seconds clock
                # (the original approach here) instead let simultaneously-
                # blended clips of different authored lengths drift to
                # unrelated points in their stride, producing an
                # incoherent pose.
                if dominant_clip is not None and dominant_clip.duration > 0.0:
                    obj["locomotion_phase"] = (obj["locomotion_phase"] + dt / dominant_clip.duration) % 1.0
                phase = obj["locomotion_phase"]
                weighted_lower = [
                    (name, phase * skeleton.animations[name].duration, weight)
                    for name, weight in lower_weights
                    if name in skeleton.animations and skeleton.animations[name].duration > 0.0
                ]

                # anim_time specifically must track the dominant clip's
                # own actual sampled time here (not be left stale from
                # before the blend space was entered) - set_skeletal_
                # animation freezes whatever anim_time currently holds as
                # prev_anim_time the instant it's called, and a stale
                # value there would crossfade OUT of the wrong pose.
                if dominant_name is not None:
                    obj["animation"] = dominant_name
                    obj["anim_time"] = (
                        phase * dominant_clip.duration
                        if dominant_clip is not None and dominant_clip.duration > 0.0 else 0.0
                    )
            else:
                _advance_clip_time(skeleton, obj, dt)
                weighted_lower = [] if obj["animation"] is None else [(obj["animation"], obj["anim_time"], 1.0)]

            # Crossfade progress (see set_skeletal_animation/
            # set_skeletal_locomotion) - advances every frame regardless
            # of which clip(s) are current; reaches weight 1.0 (the
            # prev_animation fields stop mattering at that point, though
            # left set rather than cleared, which is harmless) once
            # anim_blend_elapsed catches up to anim_blend_duration, or
            # immediately if that duration is 0 (the old instant-cut
            # behavior, still available on request).
            obj["anim_blend_elapsed"] = min(obj["anim_blend_elapsed"] + dt, obj["anim_blend_duration"])
            lower_blend_weight = (
                1.0 if obj["anim_blend_duration"] <= 0.0
                else obj["anim_blend_elapsed"] / obj["anim_blend_duration"]
            )

            if upper_mask is None:
                if not weighted_lower:
                    continue
                obj["bone_matrices"] = skeleton.compute_bone_matrices_multi(
                    weighted_lower,
                    prev_animation_name=obj["prev_animation"], prev_time=obj["prev_anim_time"],
                    blend_weight=lower_blend_weight,
                )
                continue

            # Upper-body track advances independently of the lower one -
            # see add_skeletal's upper_body_root_joints and Skeleton.
            # compute_blended_bone_matrices_multi. Runs even if
            # weighted_lower is empty (lower body just sits in bind pose
            # while the upper body still plays) and even if
            # obj["upper_animation"] is None with no upper_locomotion_
            # weights either (falls back to the lower track for every
            # joint, matching the single-clip behavior exactly).
            if upper_weights is not None:
                # Own independent phase clock from the lower track's own
                # (see add_skeletal's locomotion_phase docstring) - the
                # upper body's directional table may pick a different
                # dominant clip (with its own different duration) than
                # the lower body's, so sharing one phase between them
                # would just move the same desync problem up a level.
                dominant_name, dominant_weight = max(upper_weights, key=lambda nw: nw[1], default=(None, 0.0))
                dominant_clip = skeleton.animations.get(dominant_name) if dominant_name is not None else None
                if dominant_clip is not None and dominant_clip.duration > 0.0:
                    obj["upper_locomotion_phase"] = (
                        obj["upper_locomotion_phase"] + dt / dominant_clip.duration
                    ) % 1.0
                upper_phase = obj["upper_locomotion_phase"]
                weighted_upper = [
                    (name, upper_phase * skeleton.animations[name].duration, weight)
                    for name, weight in upper_weights
                    if name in skeleton.animations and skeleton.animations[name].duration > 0.0
                ]
                # See the lower-body bookkeeping above for why upper_
                # anim_time must track the dominant clip's actual sampled
                # time too, not be left stale.
                if dominant_name is not None:
                    obj["upper_animation"] = dominant_name
                    obj["upper_anim_time"] = (
                        upper_phase * dominant_clip.duration
                        if dominant_clip is not None and dominant_clip.duration > 0.0 else 0.0
                    )
            elif obj["upper_animation"] is not None:
                # Not in the upper-body blend space right now (e.g. a
                # manual set_upper_animation/set_upper_override while the
                # LOWER body still locomotes) - advance its own
                # single-clip time normally and wrap it as a 1-entry
                # weighted list, exactly equivalent to the old single-
                # clip-only upper sampling.
                _advance_clip_time(skeleton, obj, dt, "upper_animation", "upper_anim_time", "upper_anim_loop")
                weighted_upper = [(obj["upper_animation"], obj["upper_anim_time"], 1.0)]
            else:
                weighted_upper = None

            obj["upper_anim_blend_elapsed"] = min(
                obj["upper_anim_blend_elapsed"] + dt, obj["upper_anim_blend_duration"]
            )
            upper_blend_weight = (
                1.0 if obj["upper_anim_blend_duration"] <= 0.0
                else obj["upper_anim_blend_elapsed"] / obj["upper_anim_blend_duration"]
            )

            blended_offsets, offset_blend_weight = _advance_upper_offset_blend(obj, dt)

            obj["bone_matrices"] = skeleton.compute_blended_bone_matrices_multi(
                weighted_lower, weighted_upper, upper_mask,
                lower_prev_animation=obj["prev_animation"], lower_prev_time=obj["prev_anim_time"],
                lower_blend_weight=lower_blend_weight,
                upper_prev_animation=obj["upper_prev_animation"], upper_prev_time=obj["upper_prev_anim_time"],
                upper_blend_weight=upper_blend_weight,
                upper_rotation_offsets=blended_offsets,
                # Same timer as the offset blend above - both are set
                # together by set_skeletal_upper_joint_mask, see its own
                # docstring for why sharing one timer keeps them
                # synchronized.
                upper_joint_mask_prev=obj["upper_joint_mask_prev"],
                mask_blend_weight=offset_blend_weight,
            )

        # One bone-buffer upload per object per frame, AFTER every pose
        # above is final (the loop has several early `continue`s, so this
        # can't live at its end) - every shadow cascade and the color pass
        # then just re-bind it. See skeletal_shader.py's module docstring.
        for obj in self.skeletal_objects:
            upload_bone_matrices(self.ctx, obj)

    # =============================================================
    # DIRECTIONAL SHADOW PASS
    # =============================================================

    def _bind_shadow_alpha(self, obj):
        """Binds obj's own texture/alpha factors onto self.shadow_
        program - see that program's own fragment shader for why (an
        alpha-tested discard, so a MASK/BLEND caster's shadow matches
        its actual cutout shape instead of its whole mesh silhouette).
        Called once per object per shadow-cast draw - every call site
        that renders "shadow_vao" against self.shadow_program (the
        real-time cascades in _render_shadows, and both bake-time
        passes - _bake_directional_shadow_map and bake_static_
        lighting's own per-point-light shadow cube) needs this, since
        an object's alpha factors otherwise stay at whatever the LAST
        different object left the program's uniforms holding."""
        tex = obj.get("texture")
        if tex is not None and "u_texture" in self.shadow_program:
            tex.use(location=0)
            self.shadow_program["u_texture"].value = 0
        if "u_has_texture" in self.shadow_program:
            self.shadow_program["u_has_texture"].value = obj.get("has_texture", 0)
        if "u_alpha_mode" in self.shadow_program:
            self.shadow_program["u_alpha_mode"].value = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}.get(
                obj.get("alpha_mode", "OPAQUE"), 0
            )
        if "u_alpha_cutoff" in self.shadow_program:
            self.shadow_program["u_alpha_cutoff"].value = float(obj.get("alpha_cutoff", 0.5))
        if "u_base_alpha" in self.shadow_program:
            self.shadow_program["u_base_alpha"].value = float(obj.get("base_alpha", 1.0))

    def _render_shadows(self, camera):
        # Captured before _ensure_static_shadow_map (which sets its own
        # temporary viewport for its one-time bake pass and doesn't
        # restore it) - has to reflect the real screen viewport, not
        # whatever that lazy build leaves behind on the one frame it
        # actually runs.
        old_viewport = self.ctx.viewport

        self._ensure_static_shadow_map()

        self.shadow_manager.update(camera, self.light_dir)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "front"

        resolution = self.shadow_manager.resolution
        for cascade in range(self.shadow_manager.num_cascades):
            framebuffer = self.shadow_manager.framebuffers[cascade]
            light_vp = self.shadow_manager.light_mvps[cascade]

            framebuffer.use()
            self.ctx.viewport = (0, 0, resolution, resolution)
            framebuffer.clear(depth=1.0)

            # Dynamic/skeletal objects only, NOT static (see this
            # cascade set's own comment where it's constructed in
            # __init__) - static geometry's own shadow contribution now
            # comes from self._static_shadow_texture instead (combined
            # in via calculate_shadow's own max()), a single fixed map
            # built once rather than every static object being redrawn
            # into this camera-fit cascade every single frame regardless
            # of whether the camera or the static geometry actually
            # changed - confirmed as a real, avoidable per-frame cost
            # for even a simple test map.
            #
            # This single pass now also stands in for what used to be a
            # SECOND, separate cascade render here (into a "movable_
            # shadow_manager" of its own) - that second pass drew this
            # exact same dynamic/skeletal content, camera-fit the exact
            # same way, into a second texture set for no actual
            # difference (see this Scene's own shadow_manager comment in
            # __init__), so bind_frame_uniforms' own shadow_manager AND
            # movable_shadow_manager params are both fed this one
            # manager now instead.
            for obj in self.dynamic_objects:
                light_mvp = light_vp * self._get_model_matrix(obj)
                self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                self._bind_shadow_alpha(obj)
                obj["shadow_vao"].render()

            for obj in self.skeletal_objects:
                if not obj.get("cast_shadow", True):
                    continue
                light_mvp = light_vp * self._get_model_matrix(obj)
                self.skeletal_shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                bind_bone_matrices(obj)
                obj["shadow_vao"].render()

        self.ctx.screen.use()
        self.ctx.viewport = old_viewport

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

    # =============================================================
    # PBR PASS
    # =============================================================

    def _grab_scene_textures(self, camera):
        """Populates SSR's two per-frame inputs for pbr_shader.py's
        ssr_reflect (see enable_screen_space_reflections) - self.
        _ssr_color_texture (a straight GPU-side blit of self.ctx.
        screen's just-rendered opaque color - color-only now, see
        enable_screen_space_reflections' own docstring for why depth
        is deliberately NOT blit from self.ctx.screen alongside it any
        more) and self._ssr_depth_texture (populated by a real depth-
        only re-render instead - see _render_ssr_depth_prepass).
        No-op if SSR was never enabled this scene (self._ssr_fbo is
        None) - render() only calls this at all in that case to begin
        with, so this guard is just defensive.

        Called AFTER the main _render_scene pass (so there's something
        real to grab) and BEFORE _render_ssr_pass (which needs both of
        these) - see render()'s own ordering."""
        if self._ssr_fbo is None:
            return
        self.ctx.copy_framebuffer(self._ssr_fbo, self.ctx.screen)
        self._render_ssr_depth_prepass(camera)

    def _render_ssr_depth_prepass(self, camera):
        """Renders a real camera-space depth-only pass of every opaque/
        MASK object into self._ssr_depth_texture (via self.
        _ssr_depth_fbo) - what pbr_shader.py's ssr_reflect ray-marches
        against. Replaces what used to be a blit of self.ctx.screen's
        own depth buffer straight into an SSR-owned depth texture -
        confirmed via this Scene's own temporary SSR diagnostic log
        (see enable_screen_space_reflections' own docstring) that this
        blit reliably raised GL_INVALID_OPERATION on Intel integrated
        graphics: moderngl's ctx.depth_texture() always creates a 32-bit
        FLOATING-POINT depth texture, and self.ctx.screen's own depth
        buffer - never given an explicit format request by window.py's
        WindowManager - gets whatever the driver negotiates for the
        DEFAULT framebuffer, which is essentially never floating-point
        on Windows/WGL (that's practically an FBO-only feature) -
        glBlitFramebuffer requires an EXACT depth-format match, so
        Intel's (correctly spec-conformant) driver rejects the blit
        outright where Nvidia/AMD's more permissive ones didn't. A real
        render sidesteps the whole problem: this texture's format is
        entirely self-consistent (moderngl picked it, moderngl reads
        it back, no cross-framebuffer format negotiation involved at
        all), at the cost of one extra depth-only redraw of the opaque
        scene per frame - the exact same cost class as one more shadow
        cascade (see _render_shadows, which this closely mirrors:
        same shadow_program/skeletal_shadow_program, same "shadow_vao"
        per object, just the CAMERA's own view_proj instead of a
        light's light_mvp, and one fixed view instead of several
        cascades).

        Must match EXACTLY what _render_scene actually wrote into self.
        ctx.screen's own depth buffer this frame, or SSR would ray-
        march against a depth surface that doesn't match what's really
        on screen: a BLEND object (water) is excluded, same as _render_
        scene's own main loop never writing depth for one either (see
        that method's own alpha_mode comment) - including the water's
        own depth here would otherwise make its own reflection ray
        immediately self-intersect at distance ~0. A skeletal object
        with visible_in_color=False (the local player's own first-
        person-invisible body) is excluded too, for the same "match
        what _render_scene actually drew" reason - see that method's
        own skeletal loop, which skips it identically."""
        if self._ssr_depth_fbo is None:
            return

        self._ssr_depth_fbo.use()
        self._ssr_depth_fbo.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        # No culling - same reasoning as _bake_directional_shadow_map's
        # own shadow-caster pass (a thin, single-sided wall would
        # otherwise vanish from this depth-only pass entirely on
        # whichever side gets culled).
        self.ctx.disable(moderngl.CULL_FACE)

        view_proj = camera.get_projection_matrix() * camera.get_view_matrix()

        for obj in (*self.static_objects, *self.dynamic_objects):
            if obj.get("alpha_mode") == "BLEND":
                continue
            mvp = view_proj * self._get_model_matrix(obj)
            self.shadow_program["u_light_mvp"].write(mvp.to_bytes())
            self._bind_shadow_alpha(obj)
            obj["shadow_vao"].render()

        for obj in self.skeletal_objects:
            if not obj.get("visible_in_color", True):
                continue
            mvp = view_proj * self._get_model_matrix(obj)
            self.skeletal_shadow_program["u_light_mvp"].write(mvp.to_bytes())
            bind_bone_matrices(obj)
            obj["shadow_vao"].render()

        self.ctx.screen.use()
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

    def _render_ssr_pass(self, camera):
        """Redraws ONLY the OPAQUE/MASK static objects whose material has
        water_overrides "reflection_mode": "ssr" (see add_static's own
        docstring), a second time, now with the just-grabbed scene
        color+depth (_grab_scene_textures) bound so pbr_shader.py's
        ssr_reflect can actually ray-march real geometry instead of
        falling back to the plain skybox (which is all the FIRST,
        main _render_scene pass could do - no grab exists yet that
        early). Depth func <= (not the normal <) so this redraw passes
        the depth test at the EXACT same depth the first pass already
        wrote for this exact geometry, cleanly overwriting just those
        pixels rather than needing a separate stencil/mask to target
        them. No-op if SSR was never enabled, or no static object
        actually uses it (the common case for most scenes, including
        ones that never touch this feature at all).

        BLEND objects are deliberately excluded here (unlike the OPAQUE/
        MASK ones this redraws) - they were never drawn by the main
        _render_scene pass to begin with (see its own alpha_mode
        comment), so there's no earlier opaque draw of them at this
        exact depth to overwrite; _render_transparent_objects handles
        their own SSR-reflected draw instead, later in the frame, with
        real alpha blending enabled - see that method's own comment."""
        if self._ssr_fbo is None:
            return
        ssr_objects = [
            o for o in self.static_objects
            if o.get("reflection_mode") == "ssr" and o.get("alpha_mode") != "BLEND"
        ]
        if not ssr_objects:
            return

        self.ctx.screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.CULL_FACE)

        bind_point_lights(self.pbr_program, self.point_lights)
        bind_environment(self.pbr_program, self.environment_sky_color, self.environment_ground_color)
        bind_reflection_environment(
            self.pbr_program, self.equirect_skybox_texture,
            exposure=self.equirect_exposure, is_hdr=self.equirect_is_hdr,
        )
        bind_ssr_textures(self.pbr_program, self._ssr_color_texture, self._ssr_depth_texture)
        active_shadow_manager = self.shadow_manager if ENABLE_SHADOWS else None
        view_proj = bind_frame_uniforms(
            self.pbr_program, camera, self.light_dir, active_shadow_manager,
            light_color=self.light_color, light_intensity=self.light_intensity,
            time=self._elapsed_time, near=camera.near, far=camera.far, frame=self._frame_count,
            static_shadow_texture=self._static_shadow_texture if ENABLE_SHADOWS else None,
            static_shadow_light_vp=self._static_shadow_light_vp,
        )
        for obj in ssr_objects:
            if obj.get("double_sided"):
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.cull_face = "back"
            model_matrix = self._get_model_matrix(obj)
            bind_material(self.pbr_program, obj, model_matrix, view_proj)
            obj["vao"].render()

        self.ctx.depth_func = "<"

    def _render_scene(self, camera):
        self.ctx.screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        # Point-light data and shadow textures are identical for every
        # object this frame, so bind them once here rather than inside
        # the per-object loop below. Done separately for pbr_program and
        # skeletal_program - they're two distinct compiled GL programs,
        # each with its own uniform locations, so binding one doesn't
        # affect the other even though the uniform names match.
        bind_point_lights(self.pbr_program, self.point_lights)
        bind_point_lights(self.skeletal_program, self.point_lights)
        bind_environment(self.pbr_program, self.environment_sky_color, self.environment_ground_color)
        bind_environment(self.skeletal_program, self.environment_sky_color, self.environment_ground_color)
        bind_reflection_environment(
            self.pbr_program, self.equirect_skybox_texture,
            exposure=self.equirect_exposure, is_hdr=self.equirect_is_hdr,
        )
        bind_reflection_environment(
            self.skeletal_program, self.equirect_skybox_texture,
            exposure=self.equirect_exposure, is_hdr=self.equirect_is_hdr,
        )
        # No scene grab exists yet this early in the frame (this IS the
        # pass that produces one - see _grab_scene_textures/_render_ssr_
        # pass, called AFTER this) - an "ssr" material just falls back
        # to the plain skybox reflection on THIS draw; _render_ssr_pass
        # redraws it a second time, afterward, with the real thing.
        bind_ssr_textures(self.pbr_program, None, None)
        bind_ssr_textures(self.skeletal_program, None, None)
        # view/light/shadow uniforms are identical for every object this
        # frame too (same camera, same sun, same shadow cascades) - bound
        # once per program here rather than inside the loops below (see
        # bind_frame_uniforms' own docstring). Each returns view_proj so
        # bind_material can cheaply build u_mvp per object without
        # re-deriving the camera's view/projection matrices every draw.
        active_shadow_manager = self.shadow_manager if ENABLE_SHADOWS else None
        pbr_view_proj = bind_frame_uniforms(
            self.pbr_program, camera, self.light_dir, active_shadow_manager,
            light_color=self.light_color, light_intensity=self.light_intensity,
            time=self._elapsed_time, near=camera.near, far=camera.far, frame=self._frame_count,
            static_shadow_texture=self._static_shadow_texture if ENABLE_SHADOWS else None,
            static_shadow_light_vp=self._static_shadow_light_vp,
        )
        skeletal_view_proj = bind_frame_uniforms(
            self.skeletal_program, camera, self.light_dir, active_shadow_manager,
            light_color=self.light_color, light_intensity=self.light_intensity,
            time=self._elapsed_time, near=camera.near, far=camera.far, frame=self._frame_count,
            static_shadow_texture=self._static_shadow_texture if ENABLE_SHADOWS else None,
            static_shadow_light_vp=self._static_shadow_light_vp,
        )

        # Split by alpha_mode (see model_loader.py's _extract_material/
        # Scene._load_object - "OPAQUE" is the default for anything that
        # doesn't set one): OPAQUE keeps today's exact behavior (back-
        # face culled, no blending); MASK and BLEND both render double-
        # sided (no culling) per this project's own choice to treat "not
        # fully opaque" as "render both sides" - a cutout leaf/foliage
        # card or a glass pane reads wrong lit from only one side. MASK
        # still writes depth normally (its own fragment shader `discard`s
        # below alpha_cutoff instead of blending, so it's otherwise an
        # ordinary opaque draw) and renders in this same pass as OPAQUE.
        # BLEND objects are deliberately NOT drawn here at all - see
        # _render_transparent_objects, called separately from render()
        # AFTER the skybox, and why that order specifically matters.
        # View-frustum culling (see frustum.py's own docstring for the
        # method) - built once per frame from the exact matrix already
        # being computed for rendering, so this costs nothing beyond
        # what bind_frame_uniforms already did. Objects that fail this
        # test never reach the visible/blended split below, so a culled
        # BLEND object correctly never enters _render_transparent_
        # objects' own pass either - one cull point covers both.
        frustum_planes = extract_frustum_planes(pbr_view_proj)

        blended_objects = []
        # (obj, label_prefix) rather than the plain concatenated
        # tuple render used before PROFILE_RENDER existed - only matters
        # for building each object's own profiling label below (see
        # Scene._profiled), the actual draw logic is unaffected either
        # way.
        tagged_objects = (
            *((o, "static", False) for o in self.static_objects),
            *((o, "dynamic", True) for o in self.dynamic_objects),
        )
        for obj, prefix, movable in tagged_objects:
            if not self._is_visible(obj, frustum_planes, movable=movable):
                continue
            if obj.get("alpha_mode") == "BLEND":
                # movable carried along too - _render_transparent_
                # objects needs it to depth-sort correctly (see its own
                # docstring on why draw ORDER matters now that blend
                # surfaces don't write depth).
                blended_objects.append((obj, movable))
                continue
            if obj.get("alpha_mode") == "MASK" or obj.get("double_sided"):
                # double_sided (glTF's own flag, independent of
                # alpha_mode - see model_loader.py's _build_mesh_data)
                # covers an OPAQUE material that still wants both sides
                # rendered, e.g. a water plane authored to be seen from
                # above and below - MASK already renders double-sided
                # unconditionally regardless of this flag, so either
                # condition alone is enough to skip culling.
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.cull_face = "back"
            # A lightmapped object never reads u_point_lights at all (see
            # pbr_shader.py's own u_has_lightmap branch), so the once-
            # per-frame bind above this loop is already correct/ignored
            # for it - only a movable (dynamic) object, or a static one
            # that somehow has no lightmap texture (bake_static_lighting
            # not yet run, or excluded from it), actually needs its OWN
            # nearest-lights rebind here (see _nearest_point_lights' own
            # docstring for why "nearest to THIS object" instead of
            # whatever the frame-level call left bound).
            if movable or not obj.get("lightmap_texture"):
                bind_point_lights(self.pbr_program, self._nearest_point_lights(obj["position"]))
            model_matrix = self._get_model_matrix(obj)
            bind_material(self.pbr_program, obj, model_matrix, pbr_view_proj)
            with self._profiled(f"{prefix}:{obj.get('name', '?')}"):
                obj["vao"].render()

        # Restored before the skeletal loop below - every skeletal object
        # currently renders fully opaque/back-face-culled regardless of
        # alpha_mode (skeletal_loader.py's own pipeline doesn't extract
        # one at all yet), so it needs this project's normal default
        # state, not whatever a MASK/double_sided static object mid-loop
        # left CULL_FACE disabled as.
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        for obj in self.skeletal_objects:
            if not obj.get("visible_in_color", True):
                continue
            model_matrix = self._get_model_matrix(obj)

            # Skeletal objects have no lightmapping pipeline at all (see
            # the CULL_FACE comment just above) - always real-time-lit,
            # so unlike the static/dynamic loop above this isn't
            # conditional: every skeletal object needs its OWN nearest-
            # lights rebind, not whatever the frame-level call (or the
            # previous skeletal object's own rebind) left bound.
            bind_point_lights(self.skeletal_program, self._nearest_point_lights(obj["position"]))

            # bind_material is fully generic on prog - reused as-is here
            # rather than duplicating a "bind_skeletal_material":
            # skeletal_program declares the exact same uniform names (it
            # reuses pbr_shader's fragment shader verbatim, see
            # skeletal_shader.py's docstring), so this works unmodified.
            bind_material(self.skeletal_program, obj, model_matrix, skeletal_view_proj)
            bind_bone_matrices(obj)
            with self._profiled(f"skeletal:{obj.get('name', '?')}"):
                obj["vao"].render()

        return blended_objects

    def _render_transparent_objects(self, camera, blended_objects):
        """Draws BLEND-alpha_mode objects (see _render_scene's own
        comment on why they're excluded from its own pass) - called from
        render() AFTER the skybox specifically, not just after _render_
        scene: skybox rendering fills in the framebuffer's background
        color wherever nothing opaque/cutout was drawn (see render_
        skybox/render_equirect_skybox's own depth_func<=-at-the-far-
        plane trick), and a BLEND object's own alpha blending reads
        whatever color is ALREADY in the framebuffer at that pixel to mix
        with. Draw this pass before the skybox instead, and a translucent
        surface with open sky (or, more generally, any not-yet-drawn
        background) behind it blends against the raw glClear color
        instead of the real sky/background - confirmed exactly this bug
        via a tree's leaf-card material (BLEND alpha_mode): the gaps
        between leaves showed flat background instead of the sky actually
        behind them, because the skybox hadn't been drawn yet when those
        pixels were blended. Depth WRITES from the opaque/cutout pass
        (and the skeletal pass) still correctly stop the skybox from
        drawing over real geometry either way, so drawing skybox in
        between doesn't risk painting over anything solid - only ever
        fills in pixels nothing else claimed yet, which is exactly what a
        BLEND object still sitting in front of should keep blending
        against.

        blended_objects: list of (obj, movable) pairs (see _render_
        scene's own tagged_objects) - sorted back-to-front by distance
        from `camera` before drawing (farthest first, nearest last), so
        the nearest surface's own blend correctly composites ON TOP of
        anything farther behind it. This matters specifically because
        this pass no longer writes depth (see the depth_mask comment
        below) - depth WRITES used to at least make same-pixel ordering
        between two blend surfaces partially self-correcting (whichever
        drew first "won" that pixel), but with writes off there is
        nothing stopping a farther surface drawn AFTER a nearer one from
        painting straight over it in the wrong order. Sorting by
        distance is the standard fix short of true order-independent
        transparency - not exact for two surfaces that interpenetrate
        (there's no single "farther" object at every pixel of an
        intersection), but correct for the common case of separate,
        non-intersecting translucent surfaces this project actually
        has (water, foliage)."""
        if not blended_objects:
            return

        eye = camera.position
        def _distance_sq(entry):
            obj, movable = entry
            aabb = self._get_world_aabb(obj, movable)
            if aabb is None:
                return 0.0
            center = glm.vec3(
                (aabb[0][0] + aabb[1][0]) * 0.5,
                (aabb[0][1] + aabb[1][1]) * 0.5,
                (aabb[0][2] + aabb[1][2]) * 0.5,
            )
            delta = center - eye
            return glm.dot(delta, delta)
        # Farthest first, nearest last - see this method's own docstring
        # for why draw order matters now.
        blended_objects = sorted(blended_objects, key=_distance_sq, reverse=True)

        # depth_mask=False for this whole pass - blended surfaces still
        # DEPTH TEST normally (so they correctly disappear behind real
        # opaque geometry), they just don't WRITE depth themselves.
        # Without this, the FIRST blend object drawn at a pixel writes
        # its own depth, and every blend object drawn AFTER it then
        # fails ITS depth test at that same pixel (since it's farther
        # than whatever already wrote depth there) - not just against
        # other blend surfaces, but against ITSELF on any concave/
        # multi-sided card, and confirmed as the actual cause of a
        # translucent surface intermittently showing whatever was
        # behind IT (sky, in the reported case) instead of correctly
        # blending - exactly the class of bug this project's mainmap
        # scene started hitting once it grew to many separate BLEND
        # material groups (foliage, water) that can overlap in screen
        # space, rather than the single simple tree case this pass was
        # originally built and tested against.
        #
        # This was previously believed unavailable ("moderngl doesn't
        # expose glDepthMask at all in this version") - that check was
        # against Context itself, which indeed has no such attribute;
        # it turns out to live on Framebuffer instead (confirmed via
        # moderngl 5.12.0's own Framebuffer.depth_mask, settable on
        # self.ctx.screen, the actual bound default framebuffer this
        # whole render() call draws into).
        self.ctx.screen.depth_mask = False
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # Explicitly (re-)bound here, not left to whatever _render_scene/
        # _render_ssr_pass happened to leave on self.pbr_program earlier
        # this same frame - a BLEND material (e.g. transparent water)
        # with reflection_mode="ssr" needs the real grabbed color/depth
        # bound for its own env_reflection block to do anything but fall
        # back to the cheap skybox path (see ssr_reflect's own u_has_
        # scene_grab check) - relying on _render_ssr_pass's own call to
        # have already set this correctly would silently break the
        # moment its own ssr_objects list happened to be empty (e.g. a
        # scene whose ONLY "ssr" material is this BLEND one), since that
        # method returns early without touching these uniforms at all in
        # that case.
        bind_reflection_environment(
            self.pbr_program, self.equirect_skybox_texture,
            exposure=self.equirect_exposure, is_hdr=self.equirect_is_hdr,
        )
        bind_ssr_textures(self.pbr_program, self._ssr_color_texture, self._ssr_depth_texture)
        # Called once for this whole pass, not per object - see
        # bind_frame_uniforms' own docstring (same reasoning _render_
        # scene already applies to its own static/dynamic/skeletal
        # loops). time/near/far/frame added for the same reason as the
        # reflection/SSR binding above - a BLEND water material's own
        # animated normal pan (u_time) and SSR ray march (u_near/u_far)
        # need these to be genuinely fresh for THIS pass, not whatever
        # an earlier pass happened to leave behind.
        blend_active_shadow_manager = self.shadow_manager if ENABLE_SHADOWS else None
        blend_view_proj = bind_frame_uniforms(
            self.pbr_program, camera, self.light_dir, blend_active_shadow_manager,
            light_color=self.light_color, light_intensity=self.light_intensity,
            time=self._elapsed_time, near=camera.near, far=camera.far, frame=self._frame_count,
            static_shadow_texture=self._static_shadow_texture if ENABLE_SHADOWS else None,
            static_shadow_light_vp=self._static_shadow_light_vp,
        )
        for obj, _movable in blended_objects:
            # BLEND objects are never lightmapped (see add_static's own
            # alpha_mode_overrides docstring - baking assumes opaque,
            # static-lit geometry), so unlike _render_scene's own static/
            # dynamic loop this isn't conditional on movable/lightmap -
            # every one of these always needs its own nearest-lights
            # rebind, same as the skeletal loop above.
            bind_point_lights(self.pbr_program, self._nearest_point_lights(obj["position"]))
            model_matrix = self._get_model_matrix(obj)
            bind_material(self.pbr_program, obj, model_matrix, blend_view_proj)
            with self._profiled(f"blend:{obj.get('name', '?')}"):
                obj["vao"].render()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"
        # Restored before returning - _render_shadows' own depth passes
        # (next frame) and anything else touching self.ctx.screen both
        # need depth writes back on, or their own geometry would stop
        # updating the depth buffer too.
        self.ctx.screen.depth_mask = True

    # =============================================================
    # RENDER
    # =============================================================

    def render(self, camera, prog=None):
        if ENABLE_SHADOWS:
            with self._profiled("shadows"), self._profiled_cpu("shadows"):
                self._render_shadows(camera)
        with self._profiled_cpu("main_scene"):
            blended_objects = self._render_scene(camera)

        if self.skybox_textures is not None:
            with self._profiled("skybox"), self._profiled_cpu("skybox"):
                render_skybox(
                    self.ctx, self.skybox_program, self.skybox_vao, self.skybox_textures,
                    self.skybox_average_colors, camera, edge_fade=self.skybox_edge_fade
                )

        if self.equirect_skybox_texture is not None:
            with self._profiled("equirect_skybox"), self._profiled_cpu("equirect_skybox"):
                render_equirect_skybox(
                    self.ctx, self.equirect_skybox_program, self.equirect_skybox_vao,
                    self.equirect_skybox_texture, camera, exposure=self.equirect_exposure,
                    apply_tonemap=self.equirect_is_hdr
                )

        # AFTER the skybox specifically, same reasoning SSR itself is
        # placed here for: the just-grabbed color/depth (see _grab_
        # scene_textures) needs the sky already filled in, or an SSR ray
        # that reaches open sky would grab a stale/cleared background
        # instead of an actual reflected sky. No-ops (see their own
        # docstrings) if enable_screen_space_reflections was never
        # called this scene, or no static object actually uses
        # "reflection_mode": "ssr" - free for every scene/material that
        # doesn't touch this feature at all.
        with self._profiled("ssr"), self._profiled_cpu("ssr"):
            self._grab_scene_textures(camera)
            self._render_ssr_pass(camera)

        # AFTER the skybox (and the SSR redraw above, which needs to be
        # the LAST thing to touch water's own opaque pixels before
        # anything transparent composites on top of them) - see _render_
        # transparent_objects' own docstring for exactly why the skybox
        # part of this order matters.
        with self._profiled_cpu("transparent"):
            self._render_transparent_objects(camera, blended_objects)

        # self._report_profile_window()
        # self._report_render_cpu_profile_window()

    # =============================================================
    # DESTROY
    # =============================================================

    def destroy(self):
        for obj in self.static_objects:
            self._release_object(obj)
        for obj in self.dynamic_objects:
            self._release_object(obj)
        for obj in self.skeletal_objects:
            self._release_skeletal_object(obj)

        self.point_lights.clear()
        self.sound_manager.destroy()
        self.physics.destroy()

        if self.shadow_manager is not None:
            try:
                self.shadow_manager.destroy()
            except Exception:
                pass
            self.shadow_manager = None

        # Every other GL program/VAO/VBO/texture the Scene itself owns
        # (as opposed to per-object resources, released above) follows
        # the same release-then-clear pattern, so it's just a loop.
        program_and_buffer_attrs = (
            "pbr_program", "shadow_program", "bake_program",
            "dilate_program", "dilate_quad_vao", "dilate_quad_vbo",
            "skeletal_program", "skeletal_shadow_program",
            "skybox_program", "skybox_vao", "skybox_vbo",
            "equirect_skybox_program", "equirect_skybox_texture",
            "equirect_skybox_vao", "equirect_skybox_vbo",
            "_ssr_fbo", "_ssr_depth_fbo", "_ssr_color_texture", "_ssr_depth_texture",
            "_static_shadow_texture",
        )
        for attr in program_and_buffer_attrs:
            _release(getattr(self, attr))
            setattr(self, attr, None)

        if self.skybox_textures is not None:
            for tex in self.skybox_textures:
                _release(tex)
            self.skybox_textures = None
            self.skybox_average_colors = None

        self.static_objects.clear()
        self.dynamic_objects.clear()
        self.skeletal_objects.clear()

    def _release_skeletal_object(self, obj):
        for key in ("vao", "shadow_vao", "_render_ibo", "_shadow_ibo", "texture", "bone_ubo", "_material_ubo"):
            _release(obj.get(key))

        for vbo_dict_key in ("_render_vbos", "_shadow_vbos"):
            for vbo in obj.get(vbo_dict_key, {}).values():
                _release(vbo)

    def _release_object(self, obj):
        for key in (
            "vao", "shadow_vao", "lightmap_vao", "lightmap_texture",
            "texture", "metallic_roughness_texture", "normal_texture", "_material_ubo",
        ):
            _release(obj.get(key))
