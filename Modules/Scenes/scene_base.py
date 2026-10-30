from __future__ import annotations

from Modules.settings import settings
import math
import time as _time
from pathlib import Path

import moderngl
import pygame
import glm
import numpy as np

from Modules.Audio.sound_manager import SoundManager
from Modules.Audio.footstep_materials import get_footstep_sound
from Modules.Physics.physics_world import PhysicsWorld, CollisionGroup, to_physics_vec
from Modules.Graphics import pose_batch
from Modules.Graphics.pbr_shader import (
    create_program,
    create_morph_program,
    bind_material,
    bind_transform_only,
    pack_lighting,
    pack_point_lights,
    bind_packed_lighting,
    bind_frame_uniforms,
    bind_point_lights,
    bind_probe_irradiance,
    TEX_UNIT_ALBEDO,
    bind_environment,
    bind_fog,
    _has_uniform,
    bind_reflection_environment,
    bind_ssr_textures,
    invalidate_material_ubo,
    invalidate_bind_plan,
    bind_untinted_material,
    MAX_POINT_LIGHTS,
)
from Modules.Graphics.frustum import extract_frustum_planes, aabb_outside_frustum
from Modules.Graphics.shadow_module import CascadedShadowMap
from Modules.Graphics.scene_target import SceneTarget
from Modules.Graphics.bloom import Bloom
from Modules.Window import loading_screen
from Modules.Graphics.point_shadow_module import PointShadowMap
from Modules.Graphics.gltf_lights import extract_punctual_lights
from Modules.Graphics.gltf_actors import extract_actor_markers, extract_ladder_markers, extract_spawn_markers
from Modules.Actors import create_actor
from Modules.Graphics.lightmap_baker import (
    create_bake_program,
    create_lightmap,
    bake_point_light,
    bake_directional_light,
    create_dilate_program,
    create_dilate_quad_vao,
    dilate_lightmap,
    create_blur_program,
    create_blur_quad_vao,
    blur_lightmap,
)
from Modules.Graphics import model_loader as _model_loader
from Modules.Graphics.model_loader import load_glb, load_glb_by_material, _upload_texture, _has_attribute
from Modules.Graphics.gltf_morph import load_morph_primitives
from Modules.Graphics.lightmap_uv_generator import (
    DEFAULT_RESOLUTION as _LIGHTMAP_UV_DEFAULT_RESOLUTION,
    DEFAULT_PADDING_TEXELS as _LIGHTMAP_UV_DEFAULT_PADDING_TEXELS,
)
from Modules.Graphics import lightmap_cache_io
from Modules.Graphics.packed_lightmap import create_packed_lightmap
from Modules.Graphics.light_probes import (
    build_light_probe_grid, sample_light_probe_visibility, sample_light_probe_irradiance,
)
from Modules.Graphics.skeletal_loader import (
    load_skinned_glb, create_skeletal_vao, load_animation_clips, load_hat, _extract_rotation,
)
from Modules.Graphics.skeletal_shader import (
    create_skeletal_program,
    create_skeletal_shadow_program,
    bind_bone_matrices,
    bind_dissolve,
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


# Default crossfade (seconds) between two clips on a skeletal object: quick, but hides the pop of a hard cut.
_DEFAULT_ANIM_BLEND_DURATION = 0.25


# Perf-comparison switch: False skips the real-time (movable-object) shadow pass entirely. Baked lighting is
# unaffected.
ENABLE_SHADOWS = True
# Draws the opaque static/dynamic geometry depth-only first, so the expensive colour
# shader then runs only on the nearest surface at each pixel instead of on everything
# drawn over it (see Scene._render_early_z). Costs one extra cheap draw per object.
# OFF: measured on mainmap it cut the colour pass by ~0.25 ms but the pre-pass cost
# about the same, so the net FPS was flat to slightly worse (and depth ties resolve
# differently). Worth turning on for a scene with heavy overdraw or a costlier shader.
EARLY_Z_PREPASS = False

# See Scene._render_viewmodels: fraction of the depth range a viewmodel is
# squeezed into (from the near end).
_VIEWMODEL_DEPTH_SCALE = 0.05

# The viewmodel's own near plane (a separate projection, so the world keeps its depth precision): a scoped gun
# comes right up to the camera and would clip through the world camera's 0.1 m near plane.
_VIEWMODEL_NEAR = 0.01

# pbr_shader's alpha modes, for _bind_shadow_alpha.
_SHADOW_ALPHA_MODE_TO_INT = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}

# add_static's water_overrides keys -> (object key, conversion).
_WATER_PARAMS = {
    "pan1_speed": ("normal_pan1_speed", lambda v: tuple(float(x) for x in v)),
    "pan2_speed": ("normal_pan2_speed", lambda v: tuple(float(x) for x in v)),
    "uv_scale": ("normal_uv_scale", float),
    "uv2_scale": ("normal_uv2_scale", float),
    "normal_scale": ("normal_scale", float),
    "reflection_mode": ("reflection_mode", str),
}


def _release(resource):
    """Best-effort GL release for cleanup (None and errors ignored)."""
    if resource is not None:
        try:
            resource.release()
        except Exception:
            pass


def _resolve_upper_rotation_offsets(skeleton, upper_root_indices, degrees_spec):
    """{joint_index: glm.quat} from upper_rotation_offset_degrees: a {joint name: (x, y, z)} dict (unknown names
    skipped), or one (x, y, z) applied to every joint in upper_root_indices."""
    if isinstance(degrees_spec, dict):
        offsets = {}
        for name, degrees in degrees_spec.items():
            for index in skeleton.resolve_joint_indices([name]):
                offsets[index] = glm.quat(glm.radians(glm.vec3(degrees)))
        return offsets
    offset_quat = glm.quat(glm.radians(glm.vec3(degrees_spec)))
    return {i: offset_quat for i in upper_root_indices}


def _apply_upper_rotation_offsets(obj, new_offsets, blend_duration=_DEFAULT_ANIM_BLEND_DURATION):
    """Switches obj's upper-body rotation offsets to new_offsets with a crossfade from the current ones (see
    _advance_upper_offset_blend) - e.g. widening the mask from the clavicles to Spine4 eases the old corrections
    out and the new one in, instead of popping."""
    obj["upper_rotation_offsets_prev"] = obj["upper_rotation_offsets"]
    obj["upper_rotation_offsets"] = new_offsets
    obj["upper_offset_blend_elapsed"] = 0.0
    obj["upper_offset_blend_duration"] = blend_duration


def _begin_pose_snapshot(obj, duration):
    """Starts a transition that fades FROM the pose currently on screen.
    The older per-track crossfades (prev_animation & co.) fade from a single
    clip frozen at one time, which is not what the bones were actually doing
    when leaving a blend space (several clips mixed, a rotation offset, a
    mask split...) - so the pose jumped to that clip and only then blended,
    which reads as looking one way and turning back. A snapshot of the real
    last pose has no such gap. Returns True if a snapshot was taken; the
    caller then makes its own crossfade instant so the two don't stack."""
    if duration <= 0.0:
        # A hard cut must be instant: cancel a snapshot blend still running from an earlier call (an "idle" right
        # before a one-shot "draw") - blending toward the new clip made the pistol's draw look broken at first.
        obj["pose_snap"] = None
        return False
    last = obj["skeleton"].last_pose
    if last is None:
        return False
    obj["pose_snap"] = list(last)
    obj["pose_snap_elapsed"] = 0.0
    obj["pose_snap_duration"] = duration
    return True


def _advance_clip_time(skeleton, obj, dt, anim_key="animation", time_key="anim_time", loop_key="anim_loop"):
    """Advances a single-clip track's time by dt: wrapping if obj[loop_key], else holding the last frame (a jump
    takeoff). 0 for no clip or a zero-length one."""
    clip = skeleton.animations.get(obj[anim_key])
    if clip is not None and clip.duration > 0.0:
        obj[time_key] = (
            (obj[time_key] + dt) % clip.duration if obj[loop_key]
            else min(obj[time_key] + dt, clip.duration)
        )
    else:
        obj[time_key] = 0.0


def _advance_blend(obj, dt, elapsed_key, duration_key):
    """Advances a crossfade timer by dt and returns its 0..1 weight (1 at once for a zero duration)."""
    duration = obj[duration_key]
    obj[elapsed_key] = min(obj[elapsed_key] + dt, duration)
    return 1.0 if duration <= 0.0 else obj[elapsed_key] / duration


def _advance_locomotion(skeleton, obj, weights, dt, phase_key, anim_key, time_key):
    """One blend-space track (weights: [(clip, weight)]): advances its shared 0..1 phase by dt over the dominant
    clip's duration and returns [(clip, time, weight)], every clip sampled at that same fraction of its own
    length - a shared seconds clock let clips of different lengths drift to unrelated points in their stride.
    obj[anim_key]/obj[time_key] keep tracking the dominant clip, so leaving the blend space (a jump, say)
    crossfades out of the right pose."""
    animations = skeleton.animations
    dominant_name = max(weights, key=lambda nw: nw[1], default=(None, 0.0))[0]
    dominant_clip = animations.get(dominant_name) if dominant_name is not None else None
    dominant_duration = dominant_clip.duration if dominant_clip is not None else 0.0
    if dominant_duration > 0.0:
        obj[phase_key] = (obj[phase_key] + dt / dominant_duration) % 1.0
    phase = obj[phase_key]
    if dominant_name is not None:
        obj[anim_key] = dominant_name
        obj[time_key] = phase * dominant_duration if dominant_duration > 0.0 else 0.0
    return [(name, phase * animations[name].duration, weight) for name, weight in weights
            if name in animations and animations[name].duration > 0.0]


def _advance_upper_offset_blend(obj, dt):
    """Advances the upper-body rotation-offset crossfade and returns (blended_offsets, weight); the same weight
    drives the joint-mask transition. A joint in only one of the two dicts fades to/from identity."""
    offset_blend_weight = _advance_blend(obj, dt, "upper_offset_blend_elapsed", "upper_offset_blend_duration")
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


# Called at natural points during a load (which blocks the main loop for seconds). app.py points it at the network
# manager so Steam callbacks keep running - a peer's P2P session request left unanswered that long times out.
LOAD_PUMP = None

# See Scene.update: the shortest time between recomputations of a skeletal object's pose.
POSE_UPDATE_INTERVAL = 1.0 / 120.0
_POSE_INTERVALS = (POSE_UPDATE_INTERVAL, 1.0 / 100.0, 1.0 / 60.0, 1.0 / 30.0)   # every rate update() uses


def _pump_load():
    if pygame.display.get_init():
        pygame.event.pump()      # keeps Windows treating the window as responsive
    if LOAD_PUMP is not None:
        try:
            LOAD_PUMP()
        except Exception:
            pass
    return loading_screen.tick()   # animates the loading screen if one is up; True = a frame was drawn (GL state changed)


_model_loader.progress_hook = _pump_load      # between a model's texture uploads, too


# The scene's MSAA samples - in the offscreen target, since the window itself has none. 2x, not 4x: MSAA is a real
# bandwidth cost on integrated GPUs (shared memory).
_SSR_TARGET_SAMPLES = 2


def _ssr_target_samples(ctx):
    """_SSR_TARGET_SAMPLES clamped to GL_MAX_SAMPLES. Fixed rather than read from the window's pixel format: the
    window's MSAA request sometimes falls back to none, which made the offscreen target's cost (and fps) flip
    between launches on the same machine - and the scene doesn't render into the window's buffer anyway."""
    try:
        limit = int(ctx.info.get("GL_MAX_SAMPLES", _SSR_TARGET_SAMPLES))
    except Exception:
        limit = _SSR_TARGET_SAMPLES
    return max(0, min(_SSR_TARGET_SAMPLES, limit))


class Scene:
    def __init__(self, ctx, recalculate_shadows=True):
        self.ctx = ctx

        # True: bake_static_lighting() re-bakes the lightmaps and the collision shapes are rebuilt (both saved).
        # False: the saved ones load instead (baking only if they're missing) - Boardwalk loads in ~7 s instead of
        # ~19 s. A Nuitka build always uses the shipped caches: only a source run re-bakes, to pick up map and
        # lighting edits.
        self.recalculate_shadows = recalculate_shadows and "__compiled__" not in globals()
        self.lightmap_dir = Path("Assets/Lightmaps") / self.__class__.__name__

        # World Y below which the local player dies (fell out of the map). None = no kill plane; a scene opts in.
        self.min_height = None

        self.static_objects = []
        self._static_cull = None      # see _visible_statics
        self._static_render_cache = None      # see _static_render_arrays
        self.dynamic_objects = []
        self.skeletal_objects = []
        self.actors = []   # from add_actors_from_glb (Modules/Actors), updated generically
        # The map's "character_spawn" markers ({"position": feet glm.vec3, "yaw": degrees}),
        # filled by add_spawns_from_glb; GameMode.use_scene_spawns turns them into spawn points.
        self.player_spawns = []
        self.viewmodel_objects = []   # see add_viewmodel
        self.gibs = None              # Modules/Gore GibManager, made by the game when it wants one
        self._light_cells = {}        # see _mover_lighting
        self._still_lighting = {}     # id(obj) -> (obj, position, lighting), see _mover_lighting
        self._light_positions = None  # numpy (n, 3) of point light positions, see _nearest_point_lights
        self._pose_clocks = {}        # see update(): one clock per skeletal update rate
        self._model_cache = None      # see _get_model_matrix: id(obj) -> matrix, only during render()
        # Optional callable run just before the viewmodels are drawn, so it can
        # put something behind the first-person gun and arms (a muzzle flash).
        self.before_viewmodels = None
        self.profiler = None          # Modules/Debug FrameProfiler (set by the game): per-pass timings while F9 is on
        self._attachments = []        # see attach_skeletal
        self._skeletal_view_proj = None
        self.point_lights = []
        # Point-light visibility, sampled instead of raycasting per light per frame (light_probes.py). Built by
        # bake_static_lighting; stays None without point lights.
        self.light_probe_grid = None
        self.sound_manager = SoundManager()
        self.physics = PhysicsWorld()

        self.light_dir = glm.vec3(0.5, 1.0, 0.8)   # the sun's direction (its length is ignored)
        self.light_color = glm.vec3(1.0, 1.0, 1.0)
        self.light_intensity = 2.0

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        self.pbr_program = create_program(self.ctx)
        self.morph_program = create_morph_program(self.ctx)

        self.shadow_program = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 u_light_mvp;

                in vec3 in_position;
                in vec2 in_uv;

                out vec2 v_uv;

                invariant gl_Position;   // bit-identical to pbr_shader's, for the early-Z pre-pass
                void main()
                {
                    v_uv = in_uv;
                    gl_Position = u_light_mvp * vec4(in_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                // Alpha-tested shadows: a MASK/BLEND caster (leaves, fences) discards below its cutoff like the
                // colour pass, instead of casting its whole silhouette. BLEND uses the same cutoff - a shadow is
                // all or nothing. Shared by every shadow pass (real-time cascades, sun bake, point-light cubes);
                // uniforms left unbound read 0 = never discard.
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

        # shadow_program for morph objects, so the shadow follows the current shape-key weight instead of the rest
        # pose. No alpha test (a BLEND morph primitive casts no shadow at all).
        self.morph_shadow_program = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 u_light_mvp;
                uniform float u_morph_weight;

                in vec3 in_position;
                in vec3 in_morph_delta_position;

                invariant gl_Position;
                void main()
                {
                    vec3 morphed_position = in_position + in_morph_delta_position * u_morph_weight;
                    gl_Position = u_light_mvp * vec4(morphed_position, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                void main() {}
            """
        )

        # Real-time cascades hold only movable (dynamic/skeletal) casters - a handful of small meshes. Static
        # geometry's shadow is one fixed map (_static_shadow_texture), built on the first _render_shadows (the
        # subclass hasn't added its statics yet here). It covers the whole level at once, hence 4096 vs the
        # cascades' 2048.
        self.shadow_manager = CascadedShadowMap(self.ctx)
        self._static_shadow_texture = None
        self._static_shadow_light_vp = None
        self.static_shadow_resolution = 4096
        self.bake_program = create_bake_program(self.ctx)
        # Dilation fills the padding around each lightmap chart with its edge colour, so bilinear sampling
        # doesn't bleed the cleared black in as a dark seam (lightmap_baker.dilate_lightmap).
        self.dilate_program = create_dilate_program(self.ctx)
        self.dilate_quad_vao, self.dilate_quad_vbo = create_dilate_quad_vao(self.ctx, self.dilate_program)
        self.blur_program = create_blur_program(self.ctx)
        self.blur_quad_vao, self.blur_quad_vbo = create_blur_quad_vao(self.ctx, self.blur_program)
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
        self.skybox_yaw = 0.0    # degrees the equirect skybox is turned around the vertical axis (see MainMenuScene)
        self.skybox_fov = None   # vertical FOV the equirect skybox is drawn with; None = the camera's own
        self.skybox_flat_span = None   # degrees of the panorama's height shown FLAT, unstretched; None = projected
        # Light surfaces with the skybox's own diffuse lighting (spherical harmonics from the panorama - see
        # skybox.irradiance_sh_from_equirect) instead of its sky/ground average. Off by default; a scene opts in.
        self.use_environment_sh = False
        # Source-style distance fog (see set_fog); None = no fog.
        self.fog = None
        self.environment_sh = None
        self._base_environment_sh = None
        self.equirect_skybox_vao = None
        self.equirect_skybox_vbo = None
        self.equirect_exposure = 1.0
        self.equirect_is_hdr = True

        # Hemisphere ambient from the loaded skybox (pbr_shader's u_sky_color/u_ground_color); dim grey until one
        # is. The _base_ values are the skybox's raw averages, kept so set_ambient_intensity can re-scale any time.
        self.environment_sky_color = (0.025, 0.025, 0.025)
        self.environment_ground_color = (0.025, 0.025, 0.025)
        self._base_sky_color = self.environment_sky_color
        self._base_ground_color = self.environment_ground_color
        self.ambient_intensity = 1.0

        self._elapsed_time = 0.0   # pbr_shader's u_time (see update)
        self._frame_count = 0      # pbr_shader's u_frame_offset: varies the SSR dither pattern

        # Screen-space reflections, off until enable_screen_space_reflections().
        self._ssr_fbo = None
        self._ssr_depth_fbo = None
        self._ssr_color_texture = None
        self._ssr_depth_texture = None
        self._ssr_resolution = None
        # The offscreen MSAA target every scene renders into (Modules/Graphics/scene_target.py) - the window itself
        # has no MSAA, so present()'s copy to it is the frame's one resolve. Its colour + depth are also SSR's
        # inputs. None (it couldn't be created) = drawing straight to the window, without anti-aliasing.
        # _screen_override is the framebuffer this frame's passes draw into while the target is in use.
        self._target = None
        self._target_failed = False
        self._target_checked = False   # see present(): the first resolve into the window is checked once
        self._screen_override = None
        self._ssr_on = False   # enable_screen_space_reflections() was called
        # Glow of "bloom_" meshes (Modules/Graphics/bloom.py), built on the first frame one is visible.
        self._bloom = None
        self.bloom_intensity = 1.0

    # =============================================================
    # OBJECT LOADING
    # =============================================================

    # The material fields every loaded object dict carries (key: default), copied from the loader's result.
    _MODEL_FIELDS = {
        "vbo": None,   # the position buffer: read back for bounds (culling, the sun's bake-time shadow map)
        "has_lightmap_uv": False,
        "texture": None, "metallic_roughness_texture": None, "normal_texture": None, "emissive_texture": None,
        "metallic": 0.1, "roughness": 0.5, "emissive": [0.0, 0.0, 0.0], "normal_scale": 1.0,
        "double_sided": False, "has_texture": 0, "has_metallic_roughness_texture": 0,
        "has_normal_texture": 0, "has_emissive_texture": 0,
        "alpha_mode": "OPAQUE", "alpha_cutoff": 0.5, "base_alpha": 1.0,
    }

    def _object_from_model(self, model_path, model):
        """A scene object dict from one load_glb/load_glb_by_material result. Its VAOs (colour, shadow, bake) all
        draw from the buffers listed in "_buffers", which the object owns (see _release_object)."""
        obj = {"name": Path(model_path).stem, "vao": model["vao"], "shadow_vao": model.get("shadow_vao"),
               "lightmap_vao": model.get("lightmap_vao"), "_buffers": model.get("buffers", []),
               # set by bake_static_lighting (TEX_UNIT_SUN_LIGHTMAP in pbr_shader.py for the sun one)
               "lightmap_texture": None, "sun_lightmap_texture": None}
        for key, default in self._MODEL_FIELDS.items():
            obj[key] = model.get(key, default)
        return obj

    def _load_object(self, model_path):
        """One merged object for a (single-material) dynamic prop. The shadow pass reuses the colour pass's
        textures at draw time (_bind_shadow_alpha); no bake VAO, since only static objects are baked."""
        _pump_load()
        model = load_glb(model_path, self.ctx, self.pbr_program, extra_programs={"shadow_vao": self.shadow_program})
        return self._object_from_model(model_path, model) if model is not None else None

    def _load_objects_by_material(self, model_path):
        """One object per material in model_path (load_glb_by_material: merging a multi-material level into one
        mesh kept only the first material). add_static adds them as independent objects sharing one transform, so
        nothing downstream needs to know they came from one file. Empty list on failure."""
        _pump_load()
        groups = load_glb_by_material(model_path, self.ctx, self.pbr_program, extra_programs={
            "shadow_vao": self.shadow_program, "lightmap_vao": self.bake_program})
        if not any(m.get("has_lightmap_uv") for m in groups):
            for m in groups:   # nothing to bake into
                _release(m.pop("lightmap_vao", None))
        results = []
        for model in groups:
            obj = self._object_from_model(model_path, model)
            # material_name: what add_static's *_overrides match on.
            obj["material_name"] = model.get("material_name")
            # A "bloom_" node (Modules/Graphics/bloom.py): its index/uv buffers are exposed so the bloom pass can
            # draw it with its own position-only program (they're still owned - and freed - via "_buffers").
            obj["bloom"] = model.get("bloom", False)
            obj["bloom_name"] = (model.get("node_names") or [None])[0] if obj["bloom"] else None
            obj["bloom_ibo"] = model.get("ibo") if obj["bloom"] else None
            obj["bloom_uv_vbo"] = model.get("uv_vbo") if obj["bloom"] else None
            results.append(obj)
        return results

    # =============================================================
    # STATIC OBJECTS
    # =============================================================

    def add_static(self, model_path, position=None, rotation=None, scale=None,
                    transform=None, metallic=None, roughness=None,
                    collision=False, collision_shape="mesh", collision_mask=CollisionGroup.ALL,
                    collision_exclude_local_bounds=None, physical_material=None,
                    alpha_mode_overrides=None, roughness_overrides=None, ssr_depth_overrides=None,
                    specular_strength_overrides=None, water_overrides=None,
                    double_sided_overrides=None, ignore_source_double_sided=False,
                    collision_overrides=None, collision_object_overrides=None,
                    base_alpha_overrides=None, self_illum_overrides=None,
                    cast_shadow_overrides=None, bloom_overrides=None):
        """Adds model_path's level geometry, one static object per material (see _load_objects_by_material), and
        returns that list (empty on failure).

        collision=True gives it a collider: collision_shape "mesh" (exact triangles - level geometry) or "box"
        (from the bounds). collision_mask: see CollisionGroup. collision_exclude_local_bounds ("mesh" only) carves a
        region out (PhysicsWorld.add_static_mesh). physical_material: the footstep material ("dirt", "concrete"...,
        Modules/Audio/footstep_materials.py) for the whole collider; None = DEFAULT_FOOTSTEP_MATERIAL.

        Per-material/per-object overrides - keyed by glTF material name (model_loader._material_key) unless noted:
          collision_overrides: {material: footstep material, or None for no collider}. Usually you want
            collision_object_overrides instead - levels reuse a few trim materials across many objects.
          collision_object_overrides: {node name (Blender's object name): footstep material or None}. Wins outright
            over collision_overrides (they're not merged). Both are "mesh"-only.
          alpha_mode_overrides: {material: "OPAQUE" | "MASK" | "BLEND"}. Mainly for MASK: Blender exports a hard
            cutout (foliage, fences) as BLEND, which can't sort overlapping geometry within one object; MASK draws in
            the opaque pass instead. Only for alpha that's really binary.
          ssr_depth_overrides: {material: False} leaves it out of the SSR depth pre-pass (reflections march past it)
            - for something expensive to draw that rarely shows in reflections, like a dense tree.
          roughness_overrides: {material: float}, applied after the blanket `roughness`.
          base_alpha_overrides: {material: 0..1}. Only visible together with alpha_mode "BLEND".
          specular_strength_overrides: {material: float} - Source's $phongboost. A near-mirror material (water) in a
            dimly sun-lit scene needs it well above 1 to read as shiny at all (no Fresnel term, F0 = 0.04).
          water_overrides: {material: {...}}, any of: "pan1_speed"/"pan2_speed" (u, v) UV units/s - two panned
            samples of the normal map, like Source's water; "uv_scale" (both layers; bigger = smaller ripples);
            "uv2_scale" (layer 2 relative to layer 1 - a non-integer breaks up the repeat); "normal_scale" (ripple
            strength, overriding the file's); "reflection_mode": "cheap" (skybox, default) or "ssr" (needs
            enable_screen_space_reflections). Panning needs the material to have a normal map.
          self_illum_overrides: {material: True} - Source's $selfillum: the base texture's alpha becomes a glow
            mask (OPAQUE materials only - their alpha is otherwise unused) and the file's own emissive is cleared.
          double_sided_overrides: {material: bool} forces back-face culling off/on for an OPAQUE material (MASK and
            BLEND are always double-sided). ignore_source_double_sided=True culls every OPAQUE material not named
            there - Blender exports materials double-sided unless backface culling was ticked.
          cast_shadow_overrides: {material: False} - casts no shadow (sun map or baked point lights), e.g. a light
            fixture shadowing its own light; it still receives shadows. Needs a re-bake.
          bloom_overrides: {node name ("bloom_sphere"): {"intensity": float, "radius": float}} for the file's bloom_
            objects (Modules/Graphics/bloom.py); both default 1.0, radius 2.0 = about twice as wide."""
        models = self._load_objects_by_material(model_path)
        if not models:
            return []

        # (per-material overrides, object key, conversion), applied in this order (after the blanket values).
        material_overrides = (
            (alpha_mode_overrides, "alpha_mode", str),
            (roughness_overrides, "roughness", float),
            (ssr_depth_overrides, "ssr_depth", bool),
            (base_alpha_overrides, "base_alpha", float),
            (specular_strength_overrides, "specular_strength", float),
            (cast_shadow_overrides, "cast_shadow", bool),
            (self_illum_overrides, "self_illum", bool),
            (double_sided_overrides, "double_sided", bool),
        )
        for model in models:
            name = model.get("material_name")
            if metallic is not None:
                model["metallic"] = metallic
            if roughness is not None:
                model["roughness"] = roughness
            for overrides, key, convert in material_overrides:
                if overrides and name in overrides:
                    model[key] = convert(overrides[name])
            if model.get("self_illum"):   # self-illum is the material's one emissive source
                model["emissive"] = [0.0, 0.0, 0.0]
                model["has_emissive_texture"] = 0
            if ignore_source_double_sided and not (double_sided_overrides and name in double_sided_overrides) \
                    and model.get("alpha_mode") == "OPAQUE":   # MASK/BLEND are double-sided regardless
                model["double_sided"] = False
            if water_overrides and name in water_overrides:
                for param, value in water_overrides[name].items():
                    if param in _WATER_PARAMS:
                        key, convert = _WATER_PARAMS[param]
                        model[key] = convert(value)
            if model.get("bloom"):
                params = (bloom_overrides or {}).get(model.get("bloom_name"), {})
                model["bloom_intensity"] = float(params.get("intensity", 1.0))
                model["bloom_radius"] = max(0.1, float(params.get("radius", 1.0)))

            if transform is not None:
                model["transform"] = glm.mat4(transform)
            else:
                model["position"] = glm.vec3(position if position is not None else glm.vec3(0.0))
                model["rotation"] = glm.vec3(rotation if rotation is not None else glm.vec3(0.0))
                model["scale"] = glm.vec3(scale if scale is not None else glm.vec3(1.0))
            model["_cached_model_matrix"] = self._get_model_matrix(model)   # static: built once
            self.static_objects.append(model)

        if collision:
            # The groups are pieces of one glb sharing one transform; the collider comes from model_path itself.
            self._add_static_collision(
                model_path, models[0], collision_shape, collision_mask,
                collision_exclude_local_bounds, physical_material,
                collision_overrides, collision_object_overrides,
            )
        return models

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
                recalculate_shadows=self.recalculate_shadows,
            )
        elif collision_shape == "mesh" and collision_overrides:
            # See add_static's own collision_overrides docstring - same
            # idea, split by MATERIAL instead of by object.
            self.physics.add_static_mesh_by_material(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                material_overrides=collision_overrides, default_material=physical_material,
                recalculate_shadows=self.recalculate_shadows,
            )
        elif collision_shape == "mesh":
            # recalculate_shadows: same flag/policy Scene already uses for
            # the lightmap bake cache (bake_static_lighting) - reused here
            # for PhysicsWorld's own collision_shape_cache_io.py cache,
            # since this per-triangle Bullet mesh build turned out to be
            # an even bigger uncached load-time cost than the lightmap UV
            # generation that prompted adding that first cache (~13s vs
            # ~6s on mainmap.glb, confirmed by direct timing).
            self.physics.add_static_mesh(
                model_path, position=pos, rotation=rot, scale=scl, collision_mask=collision_mask,
                exclude_local_bounds=exclude_local_bounds, material=physical_material,
                recalculate_shadows=self.recalculate_shadows,
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

    def add_dynamic_morph(self, model_path, position=glm.vec3(0.0), rotation=glm.vec3(0.0),
                           scale=glm.vec3(1.0), physical_material=None):
        """Like add_dynamic, but for a glb with a glTF morph target
        ("shape key") - e.g. Assets/Models/Cheese/Cheesepile.glb's
        "Empty" target. Loads via gltf_morph.load_morph_primitives
        (raw accessor parsing, bypassing model_loader.py's trimesh
        pipeline entirely - trimesh has no morph target concept at all,
        see that module's own docstring for why), one dynamic_objects
        entry per mesh PRIMITIVE in the file (so a sibling primitive with
        no morph target of its own, e.g. Cheesepile's "Plate", still
        renders correctly with its own material - every primitive is
        built against self.morph_program regardless of whether it
        personally has a nonzero morph delta, at zero runtime cost for
        the ones that don't).

        Every returned object shares ONE "morph_weight" value, driven
        together via set_morph_weight(objs, weight) below - this project
        has no case yet for a model whose several primitives each need
        an INDEPENDENT weight, so this doesn't support that.

        No solid collider is built here at all (unlike add_dynamic) - a
        morphing shape's own silhouette changes constantly, so a fixed
        collision shape built from its rest pose would be wrong for most
        of the animation; add one by hand against self.physics afterward
        if a future morph-target actor actually needs to be solid.

        Shadow casting uses self.morph_shadow_program (built in Scene.
        __init__, mirroring morph_program's own relationship to pbr_
        program) - the SAME morph offset the color pass gets, driven by
        the same set_morph_weight(weight) below, so a cast shadow
        actually tracks this object's current shape-key animation
        instead of always showing its rest pose. No alpha-tested discard
        support there, unlike shadow_program - fine since every
        primitive here is OPAQUE or BLEND (a BLEND one casts no shadow at all)."""
        primitives = load_morph_primitives(model_path)
        if not primitives:
            return None

        objs = []
        for prim in primitives:
            count = len(prim["position"])
            attr_data = {
                "in_position": (prim["position"], "3f"),
                "in_normal": (prim["normal"], "3f"),
                "in_color": (np.ones((count, 3), dtype=np.float32), "3f"),
                "in_uv": (prim["uv"], "2f"),
                "in_lightmap_uv": (np.zeros((count, 2), dtype=np.float32), "2f"),
                "in_tangent": (np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1)), "4f"),
                "in_morph_delta_position": (prim["morph_delta_position"], "3f"),
                "in_morph_delta_normal": (prim["morph_delta_normal"], "3f"),
            }
            vbos = {
                name: self.ctx.buffer(data.tobytes())
                for name, (data, fmt) in attr_data.items() if _has_attribute(self.morph_program, name)
            }
            small = count <= 65536
            ibo = self.ctx.buffer(prim["indices"].astype("u2" if small else "u4").tobytes())
            vao_content = [(vbos[name], attr_data[name][1], name) for name in vbos]
            vao = self.ctx.vertex_array(self.morph_program, vao_content, ibo, index_element_size=2 if small else 4)
            # morph_shadow_program, NOT shadow_program - a morph object's
            # shadow needs the SAME morph offset its color-pass vao gets
            # (see that program's own docstring for why this is a
            # separate program, and _render_shadows' own has_morph
            # branch for where this actually gets used at draw time).
            shadow_vao = self.ctx.vertex_array(
                self.morph_shadow_program,
                [
                    (vbos["in_position"], "3f", "in_position"),
                    (vbos["in_morph_delta_position"], "3f", "in_morph_delta_position"),
                ],
                ibo, index_element_size=2 if small else 4,
            )
            texture = _upload_texture(self.ctx, prim["image"])

            obj = {
                "name": Path(model_path).stem,
                "vao": vao, "shadow_vao": shadow_vao, "lightmap_vao": None,
                "vbo": vbos.get("in_position"), "normal_vbo": vbos.get("in_normal"),
                "color_vbo": vbos.get("in_color"), "uv_vbo": vbos.get("in_uv"),
                "lightmap_uv_vbo": vbos.get("in_lightmap_uv"), "tangent_vbo": vbos.get("in_tangent"),
                "morph_position_vbo": vbos.get("in_morph_delta_position"),
                "morph_normal_vbo": vbos.get("in_morph_delta_normal"),
                "ibo": ibo,
                "has_lightmap_uv": False, "lightmap_texture": None, "sun_lightmap_texture": None,
                "texture": texture, "metallic_roughness_texture": None, "normal_texture": None,
                "emissive_texture": None,
                "metallic": prim["metallic"], "roughness": prim["roughness"],
                "emissive": [0.0, 0.0, 0.0], "normal_scale": 1.0,
                "double_sided": prim["double_sided"],
                "has_texture": 1 if texture else 0, "has_metallic_roughness_texture": 0,
                "has_normal_texture": 0, "has_emissive_texture": 0,
                "alpha_mode": prim["alpha_mode"], "alpha_cutoff": prim["alpha_cutoff"],
                "base_alpha": prim["base_color_factor"][3],
                # A BLEND primitive (glass) is drawn in _render_transparent_objects, which writes
                # no depth - and the morph shadow program has no alpha test, so it would cast a
                # solid shadow. Better none.
                "cast_shadow": prim["alpha_mode"] != "BLEND",
                "material_name": prim["material_name"],
                "has_morph": True, "morph_weight": 0.0, "morph_target_names": prim["target_names"],
                "position": glm.vec3(position), "rotation": glm.vec3(rotation), "scale": glm.vec3(scale),
                "rot_speed": 0.0,
            }
            self.dynamic_objects.append(obj)
            objs.append(obj)
        return objs

    def set_morph_weight(self, objs, weight):
        """Sets the shared shape-key weight on every object add_dynamic_
        morph returned together (see its own docstring on why they share
        one weight) - read by morph_program's own u_morph_weight uniform
        at draw time (Scene._render_scene's own morph pass). Deliberately
        NOT clamped to [0, 1] - glTF's own spec doesn't clamp a mesh's
        default weight either (Cheesepile.glb's own "Empty" target is
        authored at -1, confirmed directly), and overshooting a shape key
        past its "normal" range (negative, or past 1.0) is an ordinary,
        intentional way to exaggerate or invert its delta, not a bug to
        guard against."""
        for obj in objs:
            obj["morph_weight"] = float(weight)

    def load_prop_groups(self, model_path):
        """Loads a (possibly multi-material) glb as a list of drawable objects, one
        per material, WITHOUT adding them to the scene - see add_prop_object. For a
        moving thing the game drives itself (gibs): set each object's "transform"
        (and "position") every frame. Several copies can share the GPU data
        (dict(obj)); release the originals once, with release_prop_object."""
        return self._load_objects_by_material(model_path)

    def add_prop_object(self, obj):
        """Starts drawing (and shadow-casting) an object from load_prop_groups or a
        copy of one."""
        self.dynamic_objects.append(obj)

    def remove_prop_object(self, obj):
        try:
            self.dynamic_objects.remove(obj)
        except ValueError:
            pass

    def release_prop_object(self, obj):
        """Frees the GPU data of an ORIGINAL object from load_prop_groups."""
        self._release_object(obj)

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

    def _load_tint_mask(self, path):
        """Uploads a player-color mask image (see add_skeletal's
        tint_mask_path); None for no path."""
        if path is None:
            return None
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        texture = self.ctx.texture(img.size, 4, img.tobytes())
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.repeat_x = texture.repeat_y = True
        return texture

    def add_viewmodel(self, model_path, tint_mask_path=None, animation=None, skin_index=0, node_names=None,
                      alpha_mode_overrides=None, roughness_overrides=None):
        """Loads a first-person "viewmodel" part (arms, a gun...) glued to
        the camera: an ordinary skinned model - so Scene.update animates it
        (`animation` is the clip to start on) - that only _render_viewmodels
        ever draws (visible_in_color/cast_shadow are off, so shadows, SSR and
        the main skeletal pass all skip it). A glb holding several rigs (arms
        AND gun) loads as one part per skin_index. The caller positions it
        each frame by setting obj["transform"] (a world matrix) and obj
        ["position"] (used to pick nearby lights), and shows/hides it with
        obj["viewmodel_visible"] - a hidden part isn't even animated.
        tint_mask_path: see set_skeletal_tint. None on load failure."""
        model = self.add_skeletal(
            model_path, animation=animation, visible_in_color=False, cast_shadow=False,
            tint_mask_path=tint_mask_path, skin_index=skin_index, node_names=node_names,
        )
        if model is None:
            return None
        # Per-material overrides, same idea (and material-name keys) as
        # add_static's: alpha_mode_overrides {name: "OPAQUE"|"MASK"|"BLEND"}
        # and roughness_overrides {name: float} replace what the glb authored.
        name = model.get("material_name")
        if alpha_mode_overrides and name in alpha_mode_overrides:
            model["alpha_mode"] = alpha_mode_overrides[name]
        if roughness_overrides and name in roughness_overrides:
            model["roughness"] = float(roughness_overrides[name])
        model["viewmodel"] = True
        model["viewmodel_visible"] = False
        # Distinct from viewmodel_visible (see that flag's own docstring just above) -
        # this one ONLY affects _render_viewmodels' own draw filter, set per frame by
        # ViewModel.update once a scoped-in weapon has fully eased into its sight
        # picture (WeaponsBase.scoped) and the scope overlay takes over the screen.
        # viewmodel_visible stays True the whole time, so animation keeps updating and
        # muzzle_position()/scope_point() keep reading this object's REAL current pose
        # (still needed - firing while fully scoped should still trace from the gun's
        # actual current muzzle bone, not a frozen stale one) even while nothing is
        # actually drawn on screen.
        model["hidden_while_scoped"] = False
        model["specular_strength"] = 0
        model["transform"] = glm.mat4(1.0)
        model["position"] = glm.vec3(0.0)
        self.viewmodel_objects.append(model)
        return model

    def remove_viewmodel(self, obj):
        """Takes a viewmodel part (from add_viewmodel) out of the scene and
        frees its GPU resources."""
        if obj in self.viewmodel_objects:
            self.viewmodel_objects.remove(obj)
        self.remove_skeletal(obj)

    def remove_skeletal(self, obj):
        """Removes a skeletal object (and anything attached to it or that it's
        attached to - see attach_skeletal) and frees its GPU resources."""
        self._attachments = [a for a in self._attachments if a["child"] is not obj and a["parent"] is not obj]
        if obj in self.skeletal_objects:
            self.skeletal_objects.remove(obj)
            self._release_skeletal_object(obj)

    def attach_skeletal(self, child, parent, joint_name, local_transform=None):
        """Glues skeletal object `child` to the named joint of skeletal
        object `parent` (a weapon in a character's hand): every frame
        (see _update_attachments) child's world matrix follows that joint's
        animated pose, plus `local_transform` (a glm.mat4 in the joint's own
        frame, for offsetting/rotating/scaling the child into place). The
        joint's own scale is discarded - a rig's joints carry a unit-
        conversion scale (rat.glb's is 0.024) that would otherwise shrink the
        child. child also inherits parent's visible_in_color. Returns the
        attachment, or None for an unknown joint; undo with detach_skeletal.
        """
        joints = parent["skeleton"].joints
        index = next((i for i, j in enumerate(joints) if j.name == joint_name), None)
        if index is None:
            print(f"[Scene] attach_skeletal: no joint '{joint_name}' on '{parent.get('name', '?')}'")
            return None
        attachment = {
            "child": child, "parent": parent, "joint": index,
            "bind_world": glm.inverse(joints[index].inverse_bind_matrix),
            "local": glm.mat4(local_transform) if local_transform is not None else glm.mat4(1.0),
        }
        self._attachments.append(attachment)
        return attachment

    def joint_world_position(self, obj, joint_name, local_offset=None):
        """Where the named joint of skeletal object `obj` is in the world, in
        the pose it was last updated to (a muzzle bone, a hand...). None if obj
        has no such joint or hasn't been posed yet. local_offset: an (x, y, z)
        in the joint's own frame and units (before the rig's scale) - a point
        that rides along with the joint, e.g. a barrel tip past the muzzle bone."""
        bones = obj.get("bone_matrices")
        joints = obj["skeleton"].joints
        index = next((i for i, j in enumerate(joints) if j.name == joint_name), None)
        if index is None or not bones:
            return None
        joint_in_model = bones[index] * glm.inverse(joints[index].inverse_bind_matrix)
        if local_offset is not None:
            return glm.vec3(self._get_model_matrix(obj) * joint_in_model * glm.vec4(*local_offset, 1.0))
        return glm.vec3(self._get_model_matrix(obj) * joint_in_model[3])

    def detach_skeletal(self, child):
        self._attachments = [a for a in self._attachments if a["child"] is not child]

    def _update_attachments(self):
        """Refreshes every attached object's world matrix from its parent's
        CURRENT pose. Runs at the top of render(), not in update(): the
        caller moves a character (sets obj["position"]/["rotation"]) after
        Scene.update, so anything computed there would trail it by a frame."""
        for att in self._attachments:
            parent, child = att["parent"], att["child"]
            bones = parent.get("bone_matrices")
            if not bones:
                continue
            # The joint's frame (a decompose - not cheap) only changes when the parent's pose does.
            frame_local = att.get("_frame_local") if att.get("_bones") is bones else None
            if frame_local is None:
                world = bones[att["joint"]] * att["bind_world"]
                frame = glm.translate(glm.mat4(1.0), glm.vec3(world[3])) * glm.mat4(_extract_rotation(world))
                frame_local = frame * att["local"]
                att["_frame_local"] = frame_local
                att["_bones"] = bones
            child["transform"] = self._get_model_matrix(parent) * frame_local
            child["position"] = parent["position"]
            # NOT child["visible_in_color"] = parent's own - attach_skeletal's only caller
            # (a weapon's world model, see weapons_base.py's _attach_worldmodel) already manages
            # this itself, explicitly, at every point it actually needs to change (WeaponsBase.
            # set_active on a weapon switch, RemotePlayer._set_dead/app.py's set_local_body_
            # visible on death or the third-person toggle) - inheriting it here too meant this
            # ran every single frame and force-showed EVERY attached weapon a player has ever
            # equipped (the player's own visible_in_color is always True while alive), undoing
            # set_active(False) the instant the next frame rendered. Invisible as a bug until a
            # SECOND weapon (Gouda) got a real, attached world model at the same time as the
            # first (USP) - before that there was only ever one real one to see stuck visible.

    def add_skeletal(self, model_path, position=None, rotation=None, scale=None,
                      transform=None, animation=None, metallic=None, roughness=None,
                      emissive=None, texture_path=None, tint_mask_path=None,
                      visible_in_color=True, cast_shadow=True,
                      upper_body_root_joints=None, upper_animation=None,
                      loop=True, upper_loop=True,
                      upper_rotation_offset_degrees=(0.0, 0.0, 0.0),
                      time_scale=1.0, skin_index=0, node_names=None, material_names=None):
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
        rate. 1.0 (no change) by default.

        tint_mask_path: an RGBA image (same UV layout as the model's base
        texture) that makes the object recolorable via set_skeletal_tint:
        its alpha marks the tintable region, its RGB is the base color
        with that region desaturated. Nothing changes until a tint is set.

        skin_index: which skin to load from a glb that has several - see
        skeletal_loader.load_skinned_glb, as for node_names/material_names
        (load only those mesh nodes / only primitives with those
        materials)."""
        data = load_skinned_glb(model_path, ctx=self.ctx, time_scale=time_scale, skin_index=skin_index,
                                node_names=node_names, material_names=material_names)
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

        tint_mask_texture = self._load_tint_mask(tint_mask_path)

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
            # Optional hats (see set_skeletal_hat): source data for lazy
            # loading, a cache of built hats, and the one currently worn.
            "_hat_source": data.get("hat_source"),
            "hats": {},
            "active_hat": None,
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
            "alpha_mode": data.get("alpha_mode", "OPAQUE"),
            "material_name": data.get("material_name"),
            "metallic_roughness_texture": mr_texture,
            "tint_mask_texture": tint_mask_texture,
            "tint_color": None,
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

    def set_skeletal_hat(self, obj, name):
        """Puts the named hat (a "hat_<name>" node of the glb, e.g. "cowboy")
        on a skeletal object, or takes it off with name=None/"". Built on
        first use only - geometry, own texture and VAOs - and then just
        toggled; a hat is drawn right after the base mesh with the same bones
        (see the skeletal draw loops). Returns False for an unknown name."""
        if not name:
            obj["active_hat"] = None
            return True
        hats = obj["hats"]
        if name not in hats:
            source = obj.get("_hat_source")
            data = load_hat(source, name, self.ctx) if source else None
            if data is None:
                return False
            render = create_skeletal_vao(self.ctx, self.skeletal_program, data)
            shadow = create_skeletal_vao(self.ctx, self.skeletal_shadow_program, data)
            hats[name] = {
                "vao": render["vao"], "vbos": render["vbos"], "ibo": render["ibo"],
                "shadow_vao": shadow["vao"], "_shadow_vbos": shadow["vbos"], "_shadow_ibo": shadow["ibo"],
                "texture": data.get("texture"),
            }
        obj["active_hat"] = name
        return True

    def set_skeletal_tint(self, obj, rgb):
        """Recolors the tintable region of a skeletal object built with
        tint_mask_path (see add_skeletal) to rgb - an (r, g, b) tuple of
        0-1 floats, or None to go back to the model's own colors. A no-op
        on an object with no mask."""
        if obj.get("tint_mask_texture") is None:
            return
        new = None if rgb is None else tuple(float(c) for c in rgb)
        if obj.get("tint_color") == new:
            return
        obj["tint_color"] = new
        invalidate_material_ubo(obj)

    @staticmethod
    def _active_hat(obj):
        name = obj.get("active_hat")
        return obj["hats"].get(name) if name else None

    def _draw_hat(self, obj):
        """Color pass: the worn hat, if any, over the just-drawn base mesh -
        same program/uniforms/bones, only the albedo texture differs."""
        hat = self._active_hat(obj)
        if hat is not None:
            if hat["texture"] is not None:
                hat["texture"].use(location=TEX_UNIT_ALBEDO)
            # The hat has its own UVs, which the player-color tint mask
            # doesn't line up with.
            bind_untinted_material(self.ctx, obj)
            hat["vao"].render()

    def _draw_hat_shadow(self, obj):
        hat = self._active_hat(obj)
        if hat is not None:
            hat["shadow_vao"].render()

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
        if _begin_pose_snapshot(obj, blend_duration):
            blend_duration = 0.0
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
        if _begin_pose_snapshot(obj, blend_duration):
            blend_duration = 0.0
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
        if _begin_pose_snapshot(obj, blend_duration):
            blend_duration = 0.0
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
        if _begin_pose_snapshot(obj, blend_duration):
            blend_duration = 0.0
        _apply_upper_rotation_offsets(obj, new_offsets, blend_duration)

    def set_skeletal_upper_joint_mask(self, obj, root_joint_names, blend_duration=_DEFAULT_ANIM_BLEND_DURATION,
                                      component_space=False):
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

        root_joint_names may instead be a {joint name: weight} weightlist
        (Source's $weightlist - see Skeleton.compute_joint_weights): the
        upper clip then drives each joint by its weight, a partial weight
        blending the two. component_space blends the driven joints'
        rotations relative to the skeleton root rather than their parent
        (Source's "worldspace" layers - see Skeleton.compute_blended_bone_
        matrices).

        Takes effect on the very next Scene.update()."""
        skeleton = obj["skeleton"]
        if _begin_pose_snapshot(obj, blend_duration):
            blend_duration = 0.0
        obj["upper_joint_mask_prev"] = obj["upper_joint_mask"]
        if isinstance(root_joint_names, dict):
            obj["upper_joint_mask"] = skeleton.compute_joint_weights(root_joint_names)
            root_joint_names = [name for name, weight in root_joint_names.items() if weight > 0.0]
        else:
            obj["upper_joint_mask"] = skeleton.compute_joint_mask(root_joint_names)
        obj["upper_root_indices"] = skeleton.resolve_joint_indices(root_joint_names)
        obj["upper_component_space"] = bool(component_space)
        _apply_upper_rotation_offsets(obj, {}, blend_duration)

    def load_additional_animations(self, obj, path, rename=None, time_scale=1.0, skin_index=0):
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
        rate. 1.0 (no change) by default.

        skin_index: which rig of `path` to read (see load_animation_clips)."""
        return load_animation_clips(
            path, obj["skeleton"], rename=rename, time_scale=time_scale, skin_index=skin_index)

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

        (self.equirect_skybox_texture, self.equirect_is_hdr, sky_color, ground_color,
         environment_sh) = load_equirect_texture(self.ctx, path)
        _pump_load()      # loading the sky is a few hundred ms at the start of a scene's load
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
        self._base_environment_sh = environment_sh * exposure
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
        if self._base_environment_sh is not None:
            self.environment_sh = self._base_environment_sh * self.ambient_intensity

    def set_fog(self, color, start, end, max_density=1.0, sky_density=0.0):
        """Linear distance fog, set up the way Source's env_fog_controller does it: nothing is fogged
        closer than `start` (metres of depth from the camera), fog grows linearly to `end`, and from
        there on it is `max_density` of the way to `color`. Everything the scene draws is fogged -
        the level, props, players - except the sky and the first-person viewmodel's near range.

        color: (r, g, b), 0-255 as in Hammer's fog colour (or 0-1 floats). It is the colour seen
        at full density, so match it to the sky's horizon and distant geometry fades into the sky.
        max_density: 0-1, a ceiling on how much fog there can ever be (Source's fogmaxdensity): 1.0
        = distant things vanish completely, 0.6 = they stay faintly visible through it.
        sky_density: 0 leaves the sky untouched (Source's default - the sky is infinitely far);
        above 0 hazes the sky toward the fog colour near the horizon by that much, so a thick fog
        does not leave a clear-cut line where the fogged world meets a clear sky."""
        color = tuple(float(c) for c in color)
        if max(color) > 1.0:
            color = tuple(c / 255.0 for c in color)
        self.fog = {"color": color, "start": float(start), "end": float(end),
                    "max_density": max(0.0, min(1.0, float(max_density))),
                    "sky_density": max(0.0, min(1.0, float(sky_density)))}

    def clear_fog(self):
        """Turns fog off (the default)."""
        self.fog = None

    def _env_sh_for_shader(self):
        """The spherical-harmonic sky lighting to bind this frame, or None for the plain sky/ground
        ambient (the default - see use_environment_sh)."""
        return self.environment_sh if self.use_environment_sh else None

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

    def _ensure_target(self):
        """The offscreen target, (re)built at the window's size (after a resize too). None if it can't be made -
        then the scene draws straight to the window."""
        size = tuple(self.ctx.screen.size)
        if self._target is not None and self._target.size == size:
            return self._target
        if self._target_failed:
            return None
        if self._target is not None:
            self._target.release()
            self._target = None
        try:
            self._target = SceneTarget(self.ctx, size, _ssr_target_samples(self.ctx))
            self._ssr_resolution = size
        except Exception as e:      # an unsupported sample count etc.
            print(f"[Scene] offscreen target unavailable ({e}) - drawing straight to the window, without MSAA.")
            self._target_failed = True
        return self._target

    def enable_screen_space_reflections(self):
        """Turns on screen-space reflections for materials with water_overrides "reflection_mode": "ssr": each frame
        the opaque scene's colour and depth are grabbed (_grab_scene_textures) and the water ray-marches the
        camera's own depth along the reflected ray (pbr_shader's ssr_reflect) - so it moves correctly with the
        camera, unlike the mirrored-camera planar reflection this replaced.

        The grab normally comes from the offscreen target. Without one, a colour-only screen copy plus a
        half-resolution depth-only redraw stand in (_render_ssr_depth_prepass): the window's depth can't be blitted
        into a float depth texture on every GPU (Intel rejects the format mismatch)."""
        self._ssr_on = True
        if self._ensure_target() is not None:
            return
        if self._ssr_fbo is not None:
            self._ssr_fbo.release()
        if self._ssr_depth_fbo is not None:
            self._ssr_depth_fbo.release()
        if self._ssr_color_texture is not None:
            self._ssr_color_texture.release()
        if self._ssr_depth_texture is not None:
            self._ssr_depth_texture.release()

        size = tuple(self.ctx.screen.size)
        # Colour only (RGBA, like the window): a depth attachment would make the copy hit the format mismatch.
        self._ssr_color_texture = self.ctx.texture(size, 4)
        self._ssr_color_texture.repeat_x = self._ssr_color_texture.repeat_y = False
        self._ssr_color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._ssr_fbo = self.ctx.framebuffer(color_attachments=[self._ssr_color_texture])
        # Filled by the depth-only redraw, at half resolution: the ray march only needs coarse depth, and a quarter
        # of the pixels is a quarter of that redraw's fill cost.
        self._ssr_depth_texture = self.ctx.depth_texture((max(1, size[0] // 2), max(1, size[1] // 2)))
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
            # Stable index into self.point_lights at the time this light
            # was added - what _build_light_probe_grid's visibility array
            # is indexed by, and what _nearest_point_lights looks its
            # light back up by (see light_probes.py's own docstring).
            # Assumes point lights are only ever added before
            # bake_static_lighting runs, same assumption
            # bake_static_lighting's own point-light bake loop already
            # makes.
            "_probe_index": len(self.point_lights),
        }
        self.point_lights.append(light)
        self._still_lighting.clear()
        self._light_positions = None   # rebuilt on the next _nearest_point_lights
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

    def add_ladders_from_glb(self, model_path):
        """Registers a climbable ladder volume (PhysicsWorld.add_ladder) for every "ladder" Empty in a
        glb - see gltf_actors.extract_ladder_markers for how to author one in Blender. Returns how many."""
        markers = extract_ladder_markers(model_path)
        for marker in markers:
            self.physics.add_ladder(marker["center"], marker["rotation"], marker["half_extents"])
        return len(markers)

    def add_spawns_from_glb(self, model_path):
        """Saves every "character_spawn" marker in a glb (see gltf_actors.extract_spawn_markers)
        as a player spawn point in self.player_spawns. Returns how many."""
        self.player_spawns.extend(extract_spawn_markers(model_path))
        return len(self.player_spawns)

    def add_actors_from_glb(self, model_path):
        """Reads "actor_<id>" marker nodes out of a glb (see
        Modules/Graphics/gltf_actors.py's own module docstring for the
        naming convention, including placing more than one of the same
        actor type) and spawns a registered Modules.Actors instance at
        each one's own world transform. A marker whose id isn't a
        registered actor type (a typo, or one removed from the codebase
        since the map was authored) is skipped with a console warning
        rather than crashing map load - same tolerance add_lights_from_glb
        above already has for an unsupported light type."""
        added = []
        for marker in extract_actor_markers(model_path):
            actor = create_actor(
                marker["actor_id"], self,
                marker["position"], marker["rotation"], marker["scale"],
            )
            if actor is None:
                print(
                    f"[Scene] Skipping unknown actor id '{marker['actor_id']}' "
                    f"from {model_path} - no registered actor class for it "
                    f"(check Modules/Actors/__init__.py's own imports)."
                )
                continue
            self.actors.append(actor)
            added.append(actor)

        return added

        return added

    def _nearest_point_lights(self, position):
        """The MAX_POINT_LIGHTS point lights nearest to `position`, nearest first, for a real-time-lit draw (a mover,
        or an unbaked static). The shader's array is capped (a bigger one hit "constant register limit exceeded" on
        some GPUs); static geometry sees every light through its baked lightmap instead.

        Each light's colour is scaled by how much of it reaches `position` per the light probe grid (light_probes.py
        - interpolated, so a mover crossing behind a wall doesn't flicker the way a live raycast did). A fully
        visible light is returned as the same dict, not a copy."""
        lights = self.point_lights
        if len(lights) <= MAX_POINT_LIGHTS:
            candidates = lights
        else:
            # One numpy pass over the cached positions (Boardwalk has 126 lights; a Python sort with a key function
            # was a measurable per-frame cost).
            positions = self._light_positions
            if positions is None or len(positions) != len(lights):
                positions = self._light_positions = np.array([tuple(light["position"]) for light in lights])
            delta = positions - (position.x, position.y, position.z)
            dist = np.einsum("ij,ij->i", delta, delta)
            nearest = np.argpartition(dist, MAX_POINT_LIGHTS)[:MAX_POINT_LIGHTS]
            candidates = [lights[i] for i in nearest[np.argsort(dist[nearest])]]

        if self.light_probe_grid is None:
            return candidates

        visibility = sample_light_probe_visibility(
            self.light_probe_grid, position, len(self.point_lights)
        )

        result = []
        for light in candidates:
            factor = float(visibility[light["_probe_index"]])
            if factor >= 0.999:
                result.append(light)
            else:
                result.append({**light, "color": light["color"] * factor})
        return result

    # Every GL resource an object dict can hold, whichever loader made it (plus its "_buffers" list).
    _OBJECT_GL_KEYS = (
        "vao", "vbo", "normal_vbo", "color_vbo", "uv_vbo", "lightmap_uv_vbo",
        "tangent_vbo", "morph_position_vbo", "morph_normal_vbo",
        "ibo", "shadow_vao", "lightmap_vao", "texture",
        "metallic_roughness_texture", "normal_texture", "emissive_texture",
        "lightmap_texture", "sun_lightmap_texture", "_material_ubo",
        "_material_ubo_untinted", "tint_mask_texture",
    )

    def remove_static_object(self, obj):
        """Undoes one add_static object mid-game (an actor's destroy - see cheesepile.py): releases its GL
        resources and stops drawing it. Its collision body, if any, is the caller's (PhysicsWorld.remove_body)."""
        if obj in self.static_objects:
            self.static_objects.remove(obj)
        self._release_object(obj)

    def remove_dynamic_object(self, obj):
        """remove_static_object for an add_dynamic object. (Gibs, whose copies share GPU data, use
        remove_prop_object + release_prop_object instead.)"""
        if obj in self.dynamic_objects:
            self.dynamic_objects.remove(obj)
        self._release_object(obj)

    # =============================================================
    # 3D SOUND
    # =============================================================

    def update_audio(self, camera):
        """Call once per frame from your main loop. Sound emitters
        themselves live on self.sound_manager - use
        scene.sound_manager.add_sound(...) to add one, not a method on
        Scene (see Modules/Audio/sound_manager.py)."""
        self.sound_manager.update(camera)

    def play_footstep_sound(self, material, position, volume=1.0, character=None):
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
        or manually destroy the sound afterward.

        character: who stepped (a steam id). Each character holds ONE footstep slot (see
        SoundManager.add_sound's slot): their next step takes over the channel of their
        previous one rather than taking another from the pool, so a crowd of runners
        costs one channel each, not one per step still ringing out."""
        sound_path = get_footstep_sound(material)
        if sound_path is None:
            return None
        return self.sound_manager.add_sound(
            sound_path, position, volume=volume,
            min_distance=1.0, max_distance=12.0, loop=False,
            slot=("footstep", character) if character is not None else None,
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
            if not obj.get("cast_shadow", True):
                continue
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

    def _build_light_probe_grid(self):
        """Builds self.light_probe_grid (see light_probes.py's own
        module docstring for what it's for and how _nearest_point_lights/
        _sample_probe_irradiance use it) from every static object's OWN
        world-space AABB (not one box combined across all of them - see
        light_probes.py's GRID SCOPE section for why that distinction is
        what lets a small, disconnected structure like a tunnel get real
        probe coverage regardless of how large the rest of the map is),
        using self.physics.line_of_sight as the per-cell/per-light
        occlusion test - safe to call here because add_static's collision
        geometry is already attached to self.physics by the time a Scene
        subclass's __init__ calls bake_static_lighting (every existing
        caller adds all its static objects first).

        Called unconditionally from bake_static_lighting, including on a
        cache-hit (lightmap textures loaded straight from disk) - unlike
        the lightmap bake itself, building this grid is cheap enough
        (see light_probes.py's own docstring) that there's no need for a
        matching disk cache/staleness-check scheme; it's just rebuilt
        fresh every load."""
        self.light_probe_grid = None
        if not self.point_lights or not self.static_objects:
            return

        aabbs = []
        for obj in self.static_objects:
            aabb = self._get_world_aabb(obj, movable=False)
            if aabb is not None:
                aabbs.append(aabb)
        if not aabbs:
            return

        self.light_probe_grid = build_light_probe_grid(
            aabbs, self.point_lights, self.physics.line_of_sight,
            progress=_pump_load,      # over a second on a big map: keep the loading screen moving
        )
        self._still_lighting.clear()

    _LIGHT_CELL_REFRESH = 0.1        # seconds a shared lighting sample is reused

    _SKELETAL_LIGHT_CELL = 1.5      # metres: characters share a lighting sample within a cell
    _NEAR_CHARACTER = 6.0           # metres: inside this a character animates at full rate
    _FAR_CHARACTER = 15.0           # metres: past this a character gets cheaper everything
    _SSR_CHARACTER_DISTANCE = 10.0  # ...and stops being drawn into the reflection depth pass

    def _mover_lighting(self, obj, default_cell=None, per_object=False):
        """(nearest point lights, probe irradiance) for a real-time-lit object. A small chaotic
        mover (a gib: obj["light_cell"] = cell size in metres) shares one sample, refreshed a few
        times a second, with everything else in the same cell - so a burst of six chunks costs one
        lookup and one uniform bind instead of six of each; anything else is looked up fresh."""
        position = obj["position"]
        group = obj.get("light_group")
        if per_object:
            key = ("object", id(obj))   # this object alone, refreshed a few times a second
        elif group is not None:
            key = ("group", group)      # everything in the group is lit by one sample (a death's gibs)
        else:
            cell = obj.get("light_cell", default_cell)
            if not cell:
                # This object's own lighting, recomputed once it has moved 5 cm: a fresh lookup every frame (for a
                # prop standing still, or the gun while walking) was a measured cost, and lighting can't visibly
                # change over 5 cm (the coarse 1.5 m cells above did make the gun's lighting step).
                still = self._still_lighting.get(id(obj))
                if still is not None and still[0] is obj:
                    moved = still[1] - position
                    if glm.dot(moved, moved) < 0.0025:
                        return still[2]
                lights = self._nearest_point_lights(position)
                probe = self._sample_probe_irradiance(position)
                result = (lights, probe, pack_lighting(lights, probe))
                self._still_lighting[id(obj)] = (obj, glm.vec3(position), result)
                return result
            key =(math.floor(position.x / cell), math.floor(position.y / cell), math.floor(position.z / cell))
        entry = self._light_cells.get(key)
        if entry is None or self._elapsed_time - entry[0] > self._LIGHT_CELL_REFRESH or self._elapsed_time < entry[0]:
            lights = self._nearest_point_lights(position)
            probe = self._sample_probe_irradiance(position)
            entry = (self._elapsed_time, (lights, probe, pack_lighting(lights, probe)))
            self._light_cells[key] = entry
            if len(self._light_cells) > 64:
                self._light_cells = {key: entry}
        return entry[1]

    def _sample_probe_irradiance(self, position):
        """The combined, unlimited-light-count diffuse point-light term
        for a real-time-lit object at world-space `position` - see
        light_probes.py's module docstring's "irradiance" bullet and
        pbr_shader.py's u_probe_irradiance. (0, 0, 0) (light_probes.py's
        own no-grid default) when there's no point light in the scene at
        all, or bake_static_lighting hasn't run yet."""
        return sample_light_probe_irradiance(self.light_probe_grid, position)

    def _world_surface_area(self, obj):
        """World-space triangle surface area of a static object (its model matrix applied, so
        scale counts), read back from its position/index buffers. None if they can't be read."""
        vbo = obj.get("vbo")
        vao = obj.get("vao")
        index_buffer = getattr(vao, "index_buffer", None)
        if vbo is None or index_buffer is None:
            return None
        positions = np.frombuffer(vbo.read(), dtype="f4").reshape(-1, 3).astype("f8")
        indices = np.frombuffer(
            index_buffer.read(), dtype="u2" if vao.index_element_size == 2 else "u4"
        ).reshape(-1, 3)
        if len(positions) == 0 or len(indices) == 0:
            return None
        model_np = np.frombuffer(self._get_model_matrix(obj).to_bytes(), dtype="f4").reshape(4, 4).astype("f8")
        world = np.concatenate([positions, np.ones((len(positions), 1))], axis=1) @ model_np
        tri = world[:, :3][indices]
        return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())

    def _auto_lightmap_resolutions(self, objects, max_resolution, min_resolution, texels_per_meter=None):
        """Per-object lightmap resolution (power of two in [min_resolution, max_resolution]) so
        small objects don't get a huge map. Each object's UV atlas fills the unit square, so
        texel density ~ resolution / sqrt(surface area). By default densities are matched to the
        LARGEST object, which keeps max_resolution (its current quality); every smaller object
        scales down by sqrt(area ratio), rounded UP to a power of two. texels_per_meter instead
        sets an absolute density (resolution ~ sqrt(area) * texels_per_meter)."""
        areas = [self._world_surface_area(o) for o in objects]
        known = [a for a in areas if a]
        if not known:
            return [max_resolution] * len(objects)
        reference = max(known)
        result = []
        for area in areas:
            if not area:
                result.append(max_resolution)
                continue
            if texels_per_meter:
                wanted = math.sqrt(area) * texels_per_meter
            else:
                wanted = max_resolution * math.sqrt(area / reference)
            resolution = 1 << max(int(math.ceil(math.log2(max(wanted, 1.0)))), 0)
            result.append(int(min(max(resolution, min_resolution), max_resolution)))
        return result

    def bake_static_lighting(self, lightmap_resolution=256, point_shadow_resolution=1024,
                              directional_shadow_resolution=2048, blur_sigma=2.0,
                              auto_lightmap_resolution=False, min_lightmap_resolution=1024,
                              texels_per_meter=None):
        """Call this once, after adding all static objects and point
        lights, to bake shadow-tested point light contributions (from
        lights added with cast_shadows=True) AND the directional (sun)
        light's own shadowed contribution into each static object's
        lightmap. Lights are baked one at a time (additively blended) -
        see lightmap_baker.py's docstring for why that matters.

        auto_lightmap_resolution: True gives each object its own lightmap size, with
        lightmap_resolution as the maximum - see _auto_lightmap_resolutions. The UV generator's
        chart padding is a fixed fraction of UV space (8 texels at 2048), so keep
        min_lightmap_resolution high enough (>= 1024 = 4 texel gap) that charts don't bleed into
        each other at low sizes. Changing it re-bakes (each cache file records its own size).

        blur_sigma: a Gaussian blur (lightmap_baker.blur_lightmap, in
        texels) applied after dilation, before saving to cache - see its
        own docstring for why ("soft, blurry static lighting" was
        already the documented design goal, yet nothing blurred the bake
        at all; the sharp/noisy result this left also compressed far
        worse to disk). 0 keeps the old sharp/unblurred behavior exactly
        if some object/scene genuinely wants that.

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
        they stay real-time-unshadowed only (see add_point_light).

        Also (re)builds self.light_probe_grid (see
        _build_light_probe_grid) - unconditionally, even if every
        lightmap below turns out to be a cache hit, since a dynamic/
        skeletal object's real-time point-light visibility depends on
        this grid regardless of whether any STATIC lightmap needed
        rebaking."""
        self._build_light_probe_grid()
        # has_lightmap (one of _static_render_arrays' three cached fields - see its own
        # docstring) changes here, below, on an object that already exists in self.static_
        # objects: neither its identity nor the list's length changes, so _static_render_
        # arrays' own cache-invalidation check (keyed on those two things, like _static_cull's)
        # would never notice on its own - drop it explicitly instead, unconditionally, since
        # this whole method only ever runs rarely (once at scene setup, or an explicit rebake),
        # never per frame.
        self._static_render_cache = None

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
        # A second, separate cache file per object - see TEX_UNIT_SUN_
        # LIGHTMAP's own comment in pbr_shader.py for what this holds
        # (the sun's own isolated contribution, never combined with any
        # point light) and why it needs to exist as its own texture/
        # cache entry rather than being derivable from the combined one.
        sun_cache_paths = [
            lightmap_cache_io.lightmap_cache_path(self.lightmap_dir, f"{name}_sun")
            for name in cache_names
        ]

        # One lightmap size per eligible object: all the same unless auto_lightmap_resolution.
        if auto_lightmap_resolution:
            object_resolutions = self._auto_lightmap_resolutions(
                eligible, lightmap_resolution, min(min_lightmap_resolution, lightmap_resolution),
                texels_per_meter)
            counts = {}
            for r in object_resolutions:
                counts[r] = counts.get(r, 0) + 1
            print(f"[Scene] Auto lightmap resolution (objects per size): {dict(sorted(counts.items()))}")
        else:
            object_resolutions = [lightmap_resolution] * len(eligible)

        def _load_cache(paths):
            """Returns the loaded arrays if every cache file in `paths`
            exists AND matches the resolutions currently being requested,
            else None. This check is what stops the cache from silently
            reusing stale data when lightmap_resolution or point_shadow_
            resolution change between runs."""
            loaded = []
            for path, object_resolution in zip(paths, object_resolutions):
                _pump_load()      # reading every cache file takes over a second on a big map: keep the screen moving
                array = lightmap_cache_io.load_lightmap_cache(
                    path, object_resolution, point_shadow_resolution,
                    directional_shadow_resolution
                )
                if array is None:
                    return None
                loaded.append(array)
            return loaded

        def _texture_from_cache_array(array):
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
            # R11F_G11F_B10F (4 B/texel) when available - see packed_lightmap.py; the plain
            # RGB16F texture below is the fallback.
            packed = create_packed_lightmap(self.ctx, array)
            if packed is not None:
                return packed
            texture = self.ctx.texture(
                (array.shape[1], array.shape[0]), 3, array.tobytes(), dtype="f2"
            )
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return texture

        if not self.recalculate_shadows:
            cached_arrays = _load_cache(cache_paths)
            cached_sun_arrays = _load_cache(sun_cache_paths) if cached_arrays is not None else None
            if cached_arrays is not None and cached_sun_arrays is not None:
                for obj, array, sun_array in zip(eligible, cached_arrays, cached_sun_arrays):
                    _pump_load()      # the uploads (the driver converts each to a packed format) take seconds too
                    _release(obj.get("lightmap_texture"))
                    _release(obj.get("sun_lightmap_texture"))
                    obj["lightmap_texture"] = _texture_from_cache_array(array)
                    obj["sun_lightmap_texture"] = _texture_from_cache_array(sun_array)
                    invalidate_bind_plan(obj)

                return

            print(
                f"[Scene] recalculate_shadows is False but no valid cached "
                f"lightmaps found in {self.lightmap_dir} - baking instead."
            )

        print(f"[Scene] Baking lighting for {len(eligible)} static object(s)...")

        for obj, object_resolution in zip(eligible, object_resolutions):
            _release(obj.get("lightmap_texture"))
            _release(obj.get("sun_lightmap_texture"))
            obj["lightmap_texture"] = create_lightmap(self.ctx, object_resolution)
            obj["sun_lightmap_texture"] = create_lightmap(self.ctx, object_resolution)
            invalidate_bind_plan(obj)

        def _set_bake_gl_state():
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_func = "<="
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE      # each light is ADDED onto the lightmap
            self.ctx.disable(moderngl.CULL_FACE)

        _set_bake_gl_state()

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
                    if not obj.get("cast_shadow", True):
                        continue
                    light_mvp = light_vp * self._get_model_matrix(obj)
                    self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                    self._bind_shadow_alpha(obj)
                    obj["shadow_vao"].render()

            for obj in eligible:
                # A loading-screen redraw (it runs from here, once a frame is due) turns blending off and
                # rebinds the framebuffer: without putting the bake's state back, every later light
                # OVERWRITES the lightmap instead of adding to it and the point lights vanish from it.
                if _pump_load():
                    _set_bake_gl_state()
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
            # ALSO baked into its own separate texture, in isolation from
            # every point light baked above - see TEX_UNIT_SUN_LIGHTMAP's
            # own comment for why a moving object's shadow needs this
            # (the sun's own known contribution, never combined with a
            # point light's) to subtract back out of the combined
            # lightmap_texture correctly. bake_directional_light only
            # ever reads/writes whatever's under "lightmap_texture" on
            # the dict it's given, so a shallow copy with that one key
            # swapped is all this needs - no change to bake_directional_
            # light itself.
            bake_directional_light(
                self.ctx, self.bake_program, {**obj, "lightmap_texture": obj["sun_lightmap_texture"]},
                self._get_model_matrix(obj),
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

        # Blur BEFORE dilating: blur_lightmap only reads the interior of each chart (never its
        # edge texels, which can hold light from the wrong side of a seam) and leaves the empty
        # padding gap alone, then the dilate below extends the cleaned result outward. Done
        # before the cache-save loop below, so a cache reload gets the blurred result too.
        if blur_sigma > 0.0:
            for obj in eligible:
                blur_lightmap(self.ctx, self.blur_program, self.blur_quad_vao, obj["lightmap_texture"], sigma=blur_sigma)
                blur_lightmap(self.ctx, self.blur_program, self.blur_quad_vao, obj["sun_lightmap_texture"], sigma=blur_sigma)

        # Dilate AFTER every light (points + sun) has been baked into
        # each object's lightmap, and BEFORE the cache-save loop below -
        # so a cached reload gets the already-dilated result too,
        # without needing to redo this on every load. See lightmap_
        # baker.py's dilate_lightmap for what/why.
        #
        # iterations scaled to THIS bake's own real lightmap_resolution,
        # not dilate_lightmap's own default (which only ever matches
        # generate_lightmap_uvs' internal DEFAULT_RESOLUTION assumption,
        # 256) - see both functions' own docstrings for the full
        # reasoning. +4 texels of slack past the exact scaled-up margin,
        # matching the same "a bit more than the minimum" cushion the
        # original 6-vs-4 default already used, so two dilation fronts
        # growing toward each other from opposite edges of a chart's own
        # margin comfortably meet in the middle rather than just barely
        # touching.
        for obj, object_resolution in zip(eligible, object_resolutions):
            dilate_iterations = math.ceil(
                _LIGHTMAP_UV_DEFAULT_PADDING_TEXELS
                * (object_resolution / _LIGHTMAP_UV_DEFAULT_RESOLUTION)
            ) + 4
            dilate_lightmap(
                self.ctx, self.dilate_program, self.dilate_quad_vao, obj["lightmap_texture"],
                iterations=dilate_iterations,
            )
            dilate_lightmap(
                self.ctx, self.dilate_program, self.dilate_quad_vao, obj["sun_lightmap_texture"],
                iterations=dilate_iterations,
            )

        def _save_texture_cache(texture, path, object_resolution):
            width, height = texture.size
            array = np.frombuffer(texture.read(), dtype=np.float16).reshape(height, width, 4)
            # Drop the alpha channel before persisting - it's unused dead
            # weight here (see the load path above for why).
            lightmap_cache_io.save_lightmap_cache(
                path, array[:, :, :3], object_resolution, point_shadow_resolution,
                directional_shadow_resolution
            )
            return array[:, :, :3]

        for obj, path, sun_path, object_resolution in zip(eligible, cache_paths, sun_cache_paths, object_resolutions):
            _pump_load()          # saving and re-uploading every lightmap is seconds: keep the loading screen moving
            light_array = _save_texture_cache(obj["lightmap_texture"], path, object_resolution)
            sun_array = _save_texture_cache(obj["sun_lightmap_texture"], sun_path, object_resolution)
            # Swap the (RGBA16F, 8 B/texel) bake targets for the same compact textures a cache
            # load would give, so a freshly baked session doesn't hold 2x the VRAM of a loaded one.
            _release(obj["lightmap_texture"])
            _release(obj["sun_lightmap_texture"])
            obj["lightmap_texture"] = _texture_from_cache_array(light_array)
            obj["sun_lightmap_texture"] = _texture_from_cache_array(sun_array)
            invalidate_bind_plan(obj)

        print(f"[Scene] Baked and saved {len(eligible)} lightmap(s) to {self.lightmap_dir}")

        self.ctx.enable(moderngl.CULL_FACE)
        self._screen.use()
        self.ctx.viewport = restore_viewport
        self.ctx.depth_func = "<"
        self.ctx.cull_face = "back"

    # =============================================================
    # MODEL MATRIX
    # =============================================================

    def _get_model_matrix(self, obj):
        # Only add_static sets this: static geometry never moves, so its matrix is built once.
        cached = obj.get("_cached_model_matrix")
        if cached is not None:
            return cached

        # While a frame is being rendered nothing moves: an object's matrix (asked for by the
        # shadow cascades, the colour pass, the reflection depth pass and attachments) is built
        # once. Outside a render (a muzzle position asked for mid-update) it's always fresh.
        frame_cache = self._model_cache
        if frame_cache is not None:
            hit = frame_cache.get(id(obj))
            if hit is not None:
                return hit

        if "transform" in obj:
            model = glm.mat4(obj["transform"])
            if frame_cache is not None:
                frame_cache[id(obj)] = model
            return model

        model = glm.mat4(1.0)
        model = glm.translate(model, obj["position"])

        rotation = obj["rotation"]
        model = glm.rotate(model, rotation.x, glm.vec3(1.0, 0.0, 0.0))
        model = glm.rotate(model, rotation.y, glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, rotation.z, glm.vec3(0.0, 0.0, 1.0))

        model = glm.scale(model, obj["scale"])

        if frame_cache is not None:
            frame_cache[id(obj)] = model
        return model

    @staticmethod
    def _get_local_aabb(obj):
        """Local-space (mins, maxs) numpy arrays from the position VBO, read once and cached. None without a
        "vbo" (skeletal objects - see _skeletal_culled for how characters are culled instead)."""
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
        """World-space (mins, maxs) tuples: the local box's 8 corners through the current model matrix (correct
        under rotation), or None with no local box. Cached for a static object (movable=False) - recomputing
        hundreds of these every frame was a measured cost; a dynamic one always recomputes (there are few)."""
        # An object that tracks its own bounds (aabb_world, e.g. a gib - its manager updates it once a frame) skips
        # the numpy transform, which costs more than the draw call it guards.
        explicit = obj.get("aabb_world")
        if explicit is not None:
            return explicit
        if not movable:
            cached = obj.get("_aabb_world")
            if cached is not None:
                return cached
        else:
            # A dynamic object that hasn't moved keeps its box (the numpy transform was ~0.08 ms a frame on
            # Boardwalk, for props standing still).
            model = self._get_model_matrix(obj)
            cached = obj.get("_aabb_world_at")
            if cached is not None and cached[0] == model:
                return cached[1]

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
        # Plain float tuples: aabb_outside_frustum indexes these per plane, and numpy scalars measured slower.
        aabb = (
            tuple(float(v) for v in world_corners.min(axis=0)),
            tuple(float(v) for v in world_corners.max(axis=0)),
        )

        if not movable:
            obj["_aabb_world"] = aabb
        else:
            obj["_aabb_world_at"] = (glm.mat4(model), aabb)
        return aabb

    @staticmethod
    def _skeletal_signature(obj):
        """Everything that makes two skeletal objects' bound material identical (see the grouped
        draw in _render_scene)."""
        emissive = obj.get("emissive")
        return (
            id(obj.get("texture")), id(obj.get("normal_texture")), id(obj.get("metallic_roughness_texture")),
            id(obj.get("tint_mask_texture")), obj.get("tint_color"), obj.get("specular_strength"),
            obj.get("metallic"), obj.get("roughness"), tuple(emissive) if emissive is not None else None,
            obj.get("normal_scale"), obj.get("alpha_mode"), obj.get("base_alpha"), obj.get("double_sided"),
        )

    @staticmethod
    def _skeletal_culled(obj, frustum_planes):
        """True if a skeletal object that opted in (obj["frustum_cull"] = (half width, below, above),
        metres around its position - a character) is provably outside the camera's frustum.
        Skeletal objects have no mesh bounds of their own, so without this a player behind the
        camera was drawn (skinned, lit, textured) for nothing."""
        box = obj.get("frustum_cull")
        if box is None:
            return False
        p = obj["position"]
        half, below, above = box
        return aabb_outside_frustum(
            (p.x - half, p.y - below, p.z - half), (p.x + half, p.y + above, p.z + half), frustum_planes)

    def _static_render_arrays(self):
        """want_culling/has_lightmap (numpy bool arrays) and material_group (a list), one entry per static object,
        cached like _static_cull. These never change after add_static, and reading them by index instead of
        obj.get() in the opaque draw loop measured ~160 ns -> ~9 ns per object."""
        statics = self.static_objects
        n = len(statics)
        cache = self._static_render_cache
        if cache is None or cache[0] != n or cache[1] is not statics[-1] or cache[2] is not statics[0]:
            want_culling = np.empty(n, dtype=bool)
            has_lightmap = np.empty(n, dtype=bool)
            material_group = [None] * n
            for i, obj in enumerate(statics):
                want_culling[i] = not (obj.get("alpha_mode") == "MASK" or obj.get("double_sided"))
                has_lightmap[i] = obj.get("lightmap_texture") is not None
                material_group[i] = obj.get("material_group")
            cache = self._static_render_cache = (n, statics[-1], statics[0], want_culling, has_lightmap,
                                                  material_group)
        return cache[3], cache[4], cache[5]

    def _visible_statics(self, frustum_planes):
        """The static objects not provably outside the frustum, in load order: every box against all six planes in
        one numpy pass (boxes gathered once while the static list is unchanged; no bounds = never culled).
        Returns (objects, want_culling, has_lightmap, material_group) as parallel lists for the visible subset."""
        statics = self.static_objects
        n = len(statics)
        if n == 0:
            return [], [], [], []
        cache = self._static_cull
        if cache is None or cache[0] != n or cache[1] is not statics[-1] or cache[2] is not statics[0]:
            mins = np.full((n, 3), -np.inf)
            maxs = np.full((n, 3), np.inf)
            for i, obj in enumerate(statics):
                box = self._get_world_aabb(obj, False)
                if box is not None:
                    mins[i] = box[0]
                    maxs[i] = box[1]
            cache = self._static_cull = (n, statics[-1], statics[0], mins, maxs)
        mins, maxs = cache[3], cache[4]
        planes = np.asarray(frustum_planes)                       # (6, 4): a, b, c, d
        normals = planes[:, None, :3]                             # (6, 1, 3)
        # the box corner furthest along each plane's normal - if even that is behind the plane, the
        # box is entirely outside it
        corner = np.where(normals >= 0.0, maxs[None], mins[None])                 # (6, n, 3)
        outside = ((corner * normals).sum(axis=2) + planes[:, 3:4] < 0.0).any(axis=0)
        idx = np.flatnonzero(~outside)
        want_culling, has_lightmap, material_group = self._static_render_arrays()
        idx_list = idx.tolist()
        return (
            [statics[i] for i in idx_list],
            want_culling[idx].tolist(),
            has_lightmap[idx].tolist(),
            [material_group[i] for i in idx_list],
        )

    def _is_visible(self, obj, frustum_planes, movable=False):
        """True unless obj's world box is provably outside the frustum (no box = visible). movable: True for an
        object that can move (see _get_world_aabb)."""
        aabb = self._get_world_aabb(obj, movable)
        if aabb is None:
            return True
        return not aabb_outside_frustum(aabb[0], aabb[1], frustum_planes)

    # =============================================================
    # UPDATE
    # =============================================================

    @staticmethod
    def _lap(prof, name, since):
        """Charges the time since `since` to the profiler's sub-stage `name`; returns the new 'since'."""
        now = _time.perf_counter()
        prof.add(name, now - since)
        return now

    def update(self, dt, player_positions=(), game_mode=None):
        """player_positions ([(player_id, glm.vec3)]) and game_mode are only passed on to the actors (proximity
        triggers, mode-specific actors)."""
        # Drives animated water normals (pbr_shader's u_time); wrapped hourly so float32 never loses precision.
        self._elapsed_time = (self._elapsed_time + dt) % 3600.0
        self._frame_count = (self._frame_count + 1) % 1000

        # Step collision/physics first so this frame's dynamic-object
        # sync below (and any CharacterController.get_position() calls
        # the caller makes after this) reflect where things just moved.
        prof = self.profiler if self.profiler is not None and self.profiler.enabled else None
        t = _time.perf_counter() if prof else 0.0
        self.physics.step(dt)
        if prof:
            t = self._lap(prof, "  update: physics step", t)

        for obj in self.dynamic_objects:
            physics_body = obj.get("_physics_body")
            if physics_body is not None:
                if obj.get("_kinematic"):
                    # We drive a kinematic body (spin it, then hand Bullet the result for collision).
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

        # Actors that destroyed themselves are pruned after the loop, not removed while it iterates.
        if prof:
            t = self._lap(prof, "  update: dynamic objects", t)
        for actor in self.actors:
            actor.update(dt, player_positions, game_mode)
        if any(actor._destroyed for actor in self.actors):
            self.actors = [actor for actor in self.actors if not actor._destroyed]
        if prof:
            t = self._lap(prof, "  update: actors", t)

        pose_requests = []      # characters whose pose is computed together, after this loop
        # One clock per skeletal update rate: everything on a rate ticks on the same frames, so the
        # characters due are posed together as one batch (see pose_batch.py). (+ half a frame:
        # whichever frame lands nearest the interval ticks, so a 144 fps display still updates every
        # frame instead of every other one.)
        pose_due = {}
        for rate in _POSE_INTERVALS:
            age = self._pose_clocks.get(rate, rate) + dt
            pose_due[rate] = age + 0.5 * dt >= rate
            self._pose_clocks[rate] = 0.0 if pose_due[rate] else age
        for obj in self.skeletal_objects:
            if obj.get("viewmodel") and not obj.get("viewmodel_visible"):
                continue   # hidden first-person arms: no need to animate
            skeleton = obj["skeleton"]
            upper_mask = obj.get("upper_joint_mask")
            # {joint index: glm.quat} component-space rotations on top of the animation (e.g. a player's spine
            # bending toward where they look - PlayerModel's aim_pitch_joints), or None.
            aim = obj.get("aim_offsets")

            # The clocks below (clip time, locomotion phase, crossfade progress) advance every
            # frame, but the pose itself - sampling every bone of every blended clip, the
            # expensive part - is only recomputed POSE_UPDATE_INTERVAL apart. At the hundreds
            # of fps this game can reach an animation only changes visibly a few times per
            # 1/120 s; at or below 120 fps this always recomputes (nothing changes).
            interval = POSE_UPDATE_INTERVAL
            view_dist = obj.get("_view_dist")
            if view_dist is not None:
                # A far or off-screen character's limbs can update less often: its body still
                # moves every frame, only the pose (the expensive sampling) is coarser.
                if obj.get("_culled") or view_dist > self._FAR_CHARACTER * 2.0:
                    interval = _POSE_INTERVALS[3]       # 30 Hz
                elif view_dist > self._NEAR_CHARACTER:
                    interval = _POSE_INTERVALS[2]       # 60 Hz
                else:
                    interval = _POSE_INTERVALS[1]       # 100 Hz
            if not obj.get("visible_in_color", True) and not obj.get("viewmodel"):
                # Shadow-only (the local player's own body and gun in first person): only its
                # shadow is ever seen, so its limbs don't need the near-character rate.
                interval = max(interval, _POSE_INTERVALS[2])
            skip_pose = not pose_due[interval] and obj.get("bone_matrices") is not None
            if skip_pose and view_dist is not None and obj.get("pose_snap") is None:
                # A character between its pose ticks does no per-frame animation bookkeeping at all:
                # the time is carried over and applied in one step when its tick comes (every clock
                # in here advances linearly, so the result is the same).
                obj["_dt_debt"] = obj.get("_dt_debt", 0.0) + dt
                continue
            step = dt + obj.pop("_dt_debt", 0.0)

            # Transition-from-snapshot blend (see _begin_pose_snapshot): while
            # active, Skeleton._world_matrices fades from the captured pose to
            # whatever this frame computes, with a smoothstep ease.
            snap = obj.get("pose_snap")
            if snap is not None:
                obj["pose_snap_elapsed"] += step
                w = min(1.0, obj["pose_snap_elapsed"] / obj["pose_snap_duration"])
                if w >= 1.0:
                    obj["pose_snap"] = None
                    skeleton._pose_snap = None
                else:
                    skeleton._pose_snap = (snap, w * w * (3.0 - 2.0 * w))
            else:
                skeleton._pose_snap = None

            # Every track goes to the weighted-list sampler: a blend space (set_skeletal_locomotion) as is, a single
            # clip as a one-entry list (which the sampler fast-paths).
            lower_weights = obj.get("locomotion_weights")
            upper_weights = obj.get("upper_locomotion_weights") if upper_mask is not None else None
            if lower_weights is not None:
                weighted_lower = _advance_locomotion(skeleton, obj, lower_weights, step,
                                                     "locomotion_phase", "animation", "anim_time")
            else:
                _advance_clip_time(skeleton, obj, step)
                weighted_lower = [] if obj["animation"] is None else [(obj["animation"], obj["anim_time"], 1.0)]
            # Crossfade from prev_animation (set_skeletal_animation/set_skeletal_locomotion).
            lower_blend_weight = _advance_blend(obj, step, "anim_blend_elapsed", "anim_blend_duration")

            if upper_mask is None:
                if not weighted_lower or skip_pose:
                    continue
                # A single-pose clip (a held weapon model: duration 0) that has finished blending
                # in never changes: compute it once, not every tick.
                only = weighted_lower[0][0] if len(weighted_lower) == 1 else None
                if (only is not None and lower_blend_weight >= 1.0 and obj.get("pose_snap") is None and not aim
                        and obj.get("_static_pose") == only and obj.get("bone_matrices") is not None):
                    continue
                clip = skeleton.animations.get(only) if only is not None else None
                obj["_static_pose"] = only if (clip is not None and clip.duration <= 0.0
                                               and lower_blend_weight >= 1.0) else None
                request = pose_batch.make_request(
                    skeleton, weighted_lower, obj["prev_animation"], obj["prev_anim_time"], lower_blend_weight,
                    rotation_offsets=aim)
                if request is not None:
                    pose_requests.append((obj, request))
                    continue
                obj["bone_matrices"] = skeleton.compute_bone_matrices_multi(
                    weighted_lower,
                    prev_animation_name=obj["prev_animation"], prev_time=obj["prev_anim_time"],
                    blend_weight=lower_blend_weight, rotation_offsets=aim,
                )
                continue

            # The upper body is its own track with its own phase clock (its directional table may pick a different
            # dominant clip than the lower body's). With neither weights nor a clip it follows the lower track.
            if upper_weights is not None:
                weighted_upper = _advance_locomotion(skeleton, obj, upper_weights, step,
                                                     "upper_locomotion_phase", "upper_animation", "upper_anim_time")
            elif obj["upper_animation"] is not None:
                # A single clip (an override) while the lower body may still be locomoting.
                _advance_clip_time(skeleton, obj, step, "upper_animation", "upper_anim_time", "upper_anim_loop")
                weighted_upper = [(obj["upper_animation"], obj["upper_anim_time"], 1.0)]
            else:
                weighted_upper = None
            upper_blend_weight = _advance_blend(obj, step, "upper_anim_blend_elapsed", "upper_anim_blend_duration")

            blended_offsets, offset_blend_weight = _advance_upper_offset_blend(obj, step)
            if aim:
                # Both are component-space turns at a joint: the aim goes on top of the pose's own correction.
                merged = dict(blended_offsets or {})
                for joint, rotation in aim.items():
                    merged[joint] = rotation * merged[joint] if joint in merged else rotation
                blended_offsets = merged

            if skip_pose:
                continue
            # Batched with the other characters when it can be (see pose_batch); the per-joint weights already
            # eased if the mask is changing, as compute_blended_bone_matrices_multi would.
            component = obj.get("upper_component_space", False)
            upper = None
            if weighted_upper is not None:
                weights = upper_mask
                prev_mask = obj["upper_joint_mask_prev"]
                if prev_mask is not None and offset_blend_weight < 1.0 and prev_mask != upper_mask:
                    weights = [old + (new - old) * offset_blend_weight
                               for old, new in zip(map(float, prev_mask), map(float, upper_mask))]
                upper = (weighted_upper, obj["upper_prev_animation"], obj["upper_prev_anim_time"],
                         upper_blend_weight, weights, component)
            request = pose_batch.make_request(
                skeleton, weighted_lower, obj["prev_animation"], obj["prev_anim_time"], lower_blend_weight,
                rotation_offsets=blended_offsets, upper=upper)
            if request is not None:
                pose_requests.append((obj, request))
                continue
            obj["bone_matrices"] = skeleton.compute_blended_bone_matrices_multi(
                weighted_lower, weighted_upper, upper_mask,
                lower_prev_animation=obj["prev_animation"], lower_prev_time=obj["prev_anim_time"],
                lower_blend_weight=lower_blend_weight,
                upper_prev_animation=obj["upper_prev_animation"], upper_prev_time=obj["upper_prev_anim_time"],
                upper_blend_weight=upper_blend_weight,
                upper_rotation_offsets=blended_offsets,
                upper_joint_mask_prev=obj["upper_joint_mask_prev"],
                mask_blend_weight=offset_blend_weight,   # shares the offset crossfade's timer
                component_space=component,
            )

        if pose_requests:
            # Characters of the same model are evaluated together (pose_batch.evaluate's cost hardly
            # grows with their number); anything else in the batch list gets its own group.
            groups = {}
            for obj, request in pose_requests:
                groups.setdefault(pose_batch.structure_key(request[0]), []).append((obj, request))
            for members in groups.values():
                results = pose_batch.evaluate([request for _, request in members])
                for (obj, request), (bones, last_pose) in zip(members, results):
                    obj["bone_matrices"] = bones
                    request[0].last_pose = last_pose

        # One bone-buffer upload per object per frame, AFTER every pose
        # above is final (the loop has several early `continue`s, so this
        # can't live at its end) - every shadow cascade and the color pass
        # then just re-bind it. See skeletal_shader.py's module docstring.
        for obj in self.skeletal_objects:
            if obj.get("viewmodel") and not obj.get("viewmodel_visible"):
                continue
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
        # _has_uniform, not the bare `"name" in self.shadow_program`
        # this used to be - moderngl's Program has no __contains__, so
        # the bare form falls back to a full linear scan of every
        # uniform the program declares (see _has_uniform's own
        # docstring). This method alone showed up as the second-largest
        # single cost in a CPU profile of the whole renderer (called
        # once per shadow-casting object per cascade), almost entirely
        # from these 5 membership checks.
        tex = obj.get("texture")
        if tex is not None and _has_uniform(self.shadow_program, "u_texture"):
            tex.use(location=0)
            self.shadow_program["u_texture"].value = 0
        if _has_uniform(self.shadow_program, "u_has_texture"):
            self.shadow_program["u_has_texture"].value = obj.get("has_texture", 0)
        if _has_uniform(self.shadow_program, "u_alpha_mode"):
            self.shadow_program["u_alpha_mode"].value = _SHADOW_ALPHA_MODE_TO_INT.get(
                obj.get("alpha_mode", "OPAQUE"), 0
            )
        if _has_uniform(self.shadow_program, "u_alpha_cutoff"):
            self.shadow_program["u_alpha_cutoff"].value = float(obj.get("alpha_cutoff", 0.5))
        if _has_uniform(self.shadow_program, "u_base_alpha"):
            self.shadow_program["u_base_alpha"].value = float(obj.get("base_alpha", 1.0))

    # A small caster (a gib, a player) only needs drawing into the cascades whose view-distance
    # slice it could shadow something in - its own depth range plus how far its shadow can fall.
    _CASCADE_REACH = 4.0      # metres either side of a small caster's own depth
    _SKELETAL_SHADOW_RADIUS = 2.5

    @staticmethod
    def _in_cascade(center, radius, near_d, far_d, cam_pos, cam_forward):
        depth = glm.dot(center - cam_pos, cam_forward)
        reach = radius + Scene._CASCADE_REACH
        return depth + reach >= near_d and depth - reach <= far_d

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
        cam_pos = glm.vec3(camera.position)
        cam_forward = glm.normalize(glm.vec3(camera.front))
        casters = {}        # id(obj) -> (centre, radius) of every mover drawn into the cascades, or None
        bounds = [self.shadow_manager.near, *self.shadow_manager.splits, self.shadow_manager.far]
        for cascade in range(self.shadow_manager.num_cascades):
            framebuffer = self.shadow_manager.framebuffers[cascade]
            light_vp = self.shadow_manager.light_mvps[cascade]
            near_d, far_d = bounds[cascade], bounds[cascade + 1]
            shadow_group = None        # material_group of the previous caster: skips re-binding its alpha state

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
                if not obj.get("cast_shadow", True):
                    continue
                sphere = obj.get("shadow_sphere")     # (centre, radius) of a small mover
                if sphere is not None and not self._in_cascade(sphere[0], sphere[1], near_d, far_d, cam_pos, cam_forward):
                    continue
                self._note_caster(casters, obj, self._mover_bound(obj))
                light_mvp = light_vp * self._get_model_matrix(obj)
                if obj.get("has_morph"):
                    # Its own separate program (see morph_shadow_program's
                    # own docstring) - no alpha-tested discard to bind
                    # here, every morph object is hardcoded OPAQUE.
                    self.morph_shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                    self.morph_shadow_program["u_morph_weight"].value = obj.get("morph_weight", 0.0)
                    shadow_group = None     # next non-morph object must rebind shadow_program's own alpha state
                else:
                    self.shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                    group = obj.get("material_group")
                    if group is None or group != shadow_group:
                        self._bind_shadow_alpha(obj)
                    shadow_group = group
                obj["shadow_vao"].render()

            for obj in self.skeletal_objects:
                if not obj.get("cast_shadow", True):
                    continue
                shadow_limit = obj.get("max_shadow_distance")
                if shadow_limit is not None and obj.get("_view_dist", 0.0) > shadow_limit:
                    continue
                position = obj.get("position")
                if position is not None and not self._in_cascade(
                        glm.vec3(position), self._SKELETAL_SHADOW_RADIUS, near_d, far_d, cam_pos, cam_forward):
                    continue
                self._note_caster(casters, obj, (
                    (glm.vec3(position), self._SKELETAL_SHADOW_RADIUS) if position is not None else None))
                light_mvp = light_vp * self._get_model_matrix(obj)
                self.skeletal_shadow_program["u_light_mvp"].write(light_mvp.to_bytes())
                bind_bone_matrices(obj)
                obj["shadow_vao"].render()
                self._draw_hat_shadow(obj)

        self._screen.use()
        self.ctx.viewport = old_viewport

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        self._caster_sphere = self._union_sphere(casters)

    # The shader's soft-shadow filter, the normal offset and a character's gun/hat reach a bit
    # past a caster's own bounds.
    _CASTER_SPHERE_MARGIN = 1.0
    _radius_by_vbo = {}      # id(mesh buffer) -> (buffer, local bounding radius), see _mover_bound

    @staticmethod
    def _note_caster(casters, obj, bound):
        casters[id(obj)] = bound

    @staticmethod
    def _mover_bound(obj):
        """(centre, radius) enclosing a dynamic object however it's rotated - its local AABB's
        farthest corner from its origin, scaled - or None if it has no bounds to go on."""
        cached = obj.get("_caster_radius")
        if cached is None:
            # Keyed by the mesh buffer, so copies of one model (a burst of gibs) share one
            # answer instead of each reading the buffer back from the GPU on its first frame.
            vbo = obj.get("vbo")
            entry = Scene._radius_by_vbo.get(id(vbo)) if vbo is not None else None
            if entry is not None and entry[0] is vbo:
                cached = entry[1]
            else:
                aabb = Scene._get_local_aabb(obj)
                if aabb is None:
                    return None
                far = np.maximum(np.abs(aabb[0]), np.abs(aabb[1]))
                cached = float(np.linalg.norm(far))
                Scene._radius_by_vbo[id(vbo)] = (vbo, cached)
            obj["_caster_radius"] = cached
        scale = obj.get("scale")
        largest = max(abs(scale.x), abs(scale.y), abs(scale.z)) if scale is not None else 1.0
        return glm.vec3(obj["position"]), cached * largest

    @staticmethod
    def _union_sphere(casters):
        """One sphere enclosing every caster: None-free (see u_caster_sphere) result of
        ("none",) when nothing was drawn, ("unbounded",) if any caster has no bound, else
        ("sphere", centre, radius)."""
        if not casters:
            return ("none",)
        centre = radius = None
        for bound in casters.values():
            if bound is None:
                return ("unbounded",)
            c, r = bound
            if centre is None:
                centre, radius = glm.vec3(c), float(r)
                continue
            d = glm.distance(centre, c)
            if d + r <= radius:
                continue
            if d + radius <= r:
                centre, radius = glm.vec3(c), float(r)
                continue
            new_radius = (d + radius + r) / 2.0
            centre = centre + (glm.vec3(c) - centre) * ((new_radius - radius) / d)
            radius = new_radius
        return ("sphere", centre, radius + Scene._CASTER_SPHERE_MARGIN)

    def _bind_caster_sphere(self):
        """Uploads the frame's mover bound (see _render_shadows/_union_sphere) for the
        pbr shader's may_be_shadowed_by_movers early-out."""
        state = getattr(self, "_caster_sphere", ("unbounded",))
        prog = self.pbr_program
        if not _has_uniform(prog, "u_use_caster_sphere"):
            return
        if state[0] == "unbounded":
            prog["u_use_caster_sphere"].value = 0
            return
        prog["u_use_caster_sphere"].value = 1
        if state[0] == "none":
            prog["u_caster_sphere"].value = (0.0, 0.0, 0.0, -1.0)
        else:
            c = state[1]
            prog["u_caster_sphere"].value = (c.x, c.y, c.z, state[2])

    # =============================================================
    # PBR PASS
    # =============================================================

    @property
    def _screen(self):
        """The framebuffer the scene draws into this frame: the offscreen SceneTarget, else the window's."""
        return self._screen_override if self._screen_override is not None else self.ctx.screen

    def _ssr_enabled(self):
        return self._ssr_on

    def _ssr_textures(self):
        """(colour, depth) the reflection shader samples: the target's resolved copies, or the older grab."""
        if self._target is not None:
            return self._target.color_tex, self._target.depth_tex
        return self._ssr_color_texture, self._ssr_depth_texture

    def present(self):
        """Ends the frame's 3D drawing: resolves the offscreen target into the window and rebinds the window for the
        UI. Call after everything that draws into the scene's depth (tracers, particles) - every frame a scene is
        rendered, or nothing of it reaches the window."""
        if self._screen_override is not None:
            check = not self._target_checked
            if check:
                _ = self.ctx.error   # clears anything earlier, so the check below is about this copy alone
            self._target.present(self.ctx.screen)
            self._screen_override = None
            if check:
                self._target_checked = True
                error = self.ctx.error
                if error != "GL_NO_ERROR":
                    # This driver won't resolve into the window: draw straight to it from now on.
                    print(f"[Scene] resolving the offscreen target into the window failed ({error}) - "
                          f"drawing straight to the window instead, without MSAA.")
                    self._target.release()
                    self._target = None
                    self._target_failed = True
                    if self._ssr_on:
                        self.enable_screen_space_reflections()   # SSR's grab-and-redraw path instead
        self._screen.use()

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
        prof = self.profiler if self.profiler is not None and self.profiler.enabled else None
        t = _time.perf_counter() if prof else 0.0
        if self._screen_override is not None:
            self._target.resolve()      # colour + depth of the scene so far, copied straight from the target
            if prof:
                self._lap(prof, "  ssr: grab (resolve)", t)
            return
        if self._ssr_fbo is None:
            return
        self.ctx.copy_framebuffer(self._ssr_fbo, self.ctx.screen)
        if prof:
            t = self._lap(prof, "  ssr: grab (copy)", t)
        self._render_ssr_depth_prepass(camera)
        if prof:
            self._lap(prof, "  ssr: depth prepass", t)

    # SSR (the grab plus a full depth-only redraw of the scene) only pays for itself when the reflective
    # surface takes up a real part of the screen; a small patch of water just gets the cheap sky reflection.
    # Hysteresis (on above _ON, off below _OFF) keeps it from flickering at the threshold.
    _SSR_COVERAGE_ON = 0.015
    _SSR_COVERAGE_OFF = 0.0075

    @staticmethod
    def _screen_coverage(mins, maxs, view_proj):
        """Fraction of the screen (0..1) the world box mins..maxs covers, as its projected bounding
        rectangle. 1.0 if any corner is behind/at the camera (it can't be projected, so assume the worst)."""
        corners = np.array([[x, y, z, 1.0] for x in (mins[0], maxs[0]) for y in (mins[1], maxs[1])
                            for z in (mins[2], maxs[2])], dtype="f8")
        clip = corners @ np.array(view_proj.to_list(), dtype="f8")     # columns as rows: v @ A == M * v
        if (clip[:, 3] <= 1e-4).any():
            return 1.0
        ndc = clip[:, :2] / clip[:, 3:4]
        lo = np.clip(ndc.min(axis=0), -1.0, 1.0)
        hi = np.clip(ndc.max(axis=0), -1.0, 1.0)
        return float(max(0.0, (hi[0] - lo[0]) * (hi[1] - lo[1])) / 4.0)

    def _ssr_needed(self, obj, movable, view_proj):
        """Whether this reflective object covers enough of the screen to be worth SSR this frame."""
        aabb = self._get_world_aabb(obj, movable)
        if aabb is None:
            return True
        coverage = self._screen_coverage(aabb[0], aabb[1], view_proj)
        on = coverage >= (self._SSR_COVERAGE_OFF if obj.get("_ssr_on") else self._SSR_COVERAGE_ON)
        obj["_ssr_on"] = on
        return on

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
        # Same camera as _render_scene's own main pass, so the same
        # frustum test applied there is exactly correct here too - not
        # just an optimization shortcut: an object _render_scene culled
        # never wrote color OR depth to self.ctx.screen this frame
        # either, so redrawing it into the SSR depth texture anyway was
        # pure waste (a real, unconditional draw call - matrix build,
        # uniform writes, shadow_vao.render() - for geometry that can
        # never actually be ray-marched against, since nothing on
        # screen is behind it to reflect). Confirmed via CPU profiling
        # as a real cost: this pass previously redrew EVERY static
        # object every time SSR was active, regardless of how much of
        # the level was actually in view.
        frustum_planes = extract_frustum_planes(view_proj)

        # The main pass already found which opaque/MASK objects are in view (same camera), so
        # reuse that list instead of testing every object again.
        visible = getattr(self, "_frame_opaque_visible", None)
        if visible is None:
            # has_morph objects excluded here too, same as _render_
            # scene's own tagged_objects - their shadow_vao is built
            # against morph_shadow_program (own attributes/uniform), not
            # shadow_program, which is all this pass knows how to drive.
            visible = [
                (o, movable) for o, movable in (
                    *((o, False) for o in self.static_objects),
                    *((o, True) for o in self.dynamic_objects if not o.get("has_morph")))
                if o.get("alpha_mode") != "BLEND" and self._is_visible(o, frustum_planes, movable=movable)]
        opaque_bound = False    # the program's alpha state currently says "opaque" (no per-object texture)
        for obj, movable in visible:
            if obj.get("ssr_depth") is False:
                continue
            mvp = view_proj * self._get_model_matrix(obj)
            self.shadow_program["u_light_mvp"].write(mvp.to_bytes())
            if obj.get("alpha_mode") == "MASK":
                self._bind_shadow_alpha(obj)        # needs its own texture/cutoff
                opaque_bound = False
            elif not opaque_bound:
                self._bind_shadow_alpha(obj)        # opaque: nothing per-object, so once is enough
                opaque_bound = True
            obj["shadow_vao"].render()

        for obj in self.skeletal_objects:
            if not obj.get("visible_in_color", True) or self._skeletal_culled(obj, frustum_planes):
                continue
            if obj.get("frustum_cull") is not None and obj.get("_view_dist", 0.0) > self._SSR_CHARACTER_DISTANCE:
                continue        # too far to matter in a reflection: not worth a draw
            # No-op for a skeletal object in practice (_is_visible
            # always returns True for one - see its own docstring, no
            # cached AABB available), kept here anyway so this stays
            # correct/consistent if that ever changes rather than
            # silently relying on the exemption.
            if not self._is_visible(obj, frustum_planes, movable=True):
                continue
            mvp = view_proj * self._get_model_matrix(obj)
            self.skeletal_shadow_program["u_light_mvp"].write(mvp.to_bytes())
            bind_bone_matrices(obj)
            obj["shadow_vao"].render()
            self._draw_hat_shadow(obj)

        self._screen.use()
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

    def _render_ssr_pass(self, camera, pbr_view_proj):
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
        if not self._ssr_enabled():
            return
        ssr_objects = [
            o for o in self.static_objects
            if o.get("reflection_mode") == "ssr" and o.get("alpha_mode") != "BLEND"
        ]
        if not ssr_objects:
            return

        self._screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<="
        self.ctx.enable(moderngl.CULL_FACE)

        # Point lights/environment/reflection/camera/light/shadow
        # uniforms were already bound once this frame by _bind_pbr_
        # frame_uniforms (see its own docstring - nothing here changed
        # since then) - only the just-grabbed SSR color/depth are
        # genuinely new data this pass needs to bind itself.
        bind_ssr_textures(self.pbr_program, *self._ssr_textures())
        for obj in ssr_objects:
            if obj.get("double_sided"):
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.cull_face = "back"
            model_matrix = self._get_model_matrix(obj)
            bind_material(self.pbr_program, obj, model_matrix, pbr_view_proj)
            obj["vao"].render()

        self.ctx.depth_func = "<"

    def _bind_frame_environment(self, program, camera):
        """Binds everything a lit program needs that's the same for every object this frame (point lights,
        ambient, fog, reflection environment, camera/sun/shadow uniforms) and returns its view-projection matrix.
        Once per program per frame - bind_frame_uniforms alone was a measured cost when repeated per pass. SSR
        textures are bound empty here (no scene grab exists yet); the SSR/transparent passes rebind them."""
        bind_point_lights(program, self.point_lights, packed=self._frame_light_pack)
        bind_environment(program, self.environment_sky_color, self.environment_ground_color,
                         self._env_sh_for_shader(), self.skybox_yaw)
        bind_fog(program, self.fog)
        bind_reflection_environment(
            program, self.equirect_skybox_texture, exposure=self.equirect_exposure, is_hdr=self.equirect_is_hdr)
        bind_ssr_textures(program, None, None)
        return bind_frame_uniforms(
            program, camera, self.light_dir, self.shadow_manager if ENABLE_SHADOWS else None,
            light_color=self.light_color, light_intensity=self.light_intensity,
            time=self._elapsed_time, near=camera.near, far=camera.far, frame=self._frame_count,
            static_shadow_texture=self._static_shadow_texture if ENABLE_SHADOWS else None,
            static_shadow_light_vp=self._static_shadow_light_vp,
        )

    @staticmethod
    def _bind_lighting(program, lit):
        """Binds one _mover_lighting result: a packed buffer, or (point lights, probe irradiance)."""
        if lit[2] is not None:
            bind_packed_lighting(program, lit[2])
        else:
            bind_point_lights(program, lit[0])
            bind_probe_irradiance(program, lit[1])

    def _bind_pbr_frame_uniforms(self, camera):
        """pbr_program's per-frame state, shared by the opaque, SSR and transparent passes; returns its view-proj."""
        self._frame_light_pack = pack_point_lights(self.point_lights)   # reused by every program this frame
        return self._bind_frame_environment(self.pbr_program, camera)

    def _render_early_z(self, objects, view_proj, target=None):
        """Depth-only pre-pass of `objects` into the screen's own depth buffer, with the same
        transform and culling the colour pass will use, so _render_scene's colour draws (depth
        <=, no depth write) only shade the surface that ends up visible. Uses shadow_program's
        trivial shader; a MASK object gets its alpha cutout (see _bind_shadow_alpha) so its
        depth matches what the colour pass draws. Both vertex shaders declare gl_Position
        invariant, so the depths are bit-identical and the <= test can't punch holes.

        target: a framebuffer to draw the depth into instead of the screen (the bloom pass's
        half-resolution buffer) - it must already be bound."""
        ctx = self.ctx
        program = self.shadow_program
        screen = target if target is not None else self._screen
        screen.color_mask = (False, False, False, False)
        ctx.depth_func = "<"
        screen.depth_mask = True
        opaque_bound = False       # the program's alpha state currently says "opaque"
        for obj, _movable in objects:
            masked = obj.get("alpha_mode") == "MASK"
            if masked or obj.get("double_sided"):
                ctx.disable(moderngl.CULL_FACE)
            else:
                ctx.enable(moderngl.CULL_FACE)
                ctx.cull_face = "back"
            program["u_light_mvp"].write((view_proj * self._get_model_matrix(obj)).to_bytes())
            if masked:
                self._bind_shadow_alpha(obj)
                opaque_bound = False
            elif not opaque_bound:
                self._bind_shadow_alpha(obj)
                opaque_bound = True
            obj["shadow_vao"].render()
        screen.color_mask = (True, True, True, True)

    def _render_bloom(self, camera, view_proj):
        """The glow of this frame's visible "bloom_" objects (see Modules/Graphics/bloom.py).
        Costs nothing on a frame where none is in view."""
        if not settings.bloom:
            return
        visible = getattr(self, "_frame_opaque_visible", None) or []
        glowing = [o for o, _movable in visible if o.get("bloom")]
        if not glowing:
            return
        if self._bloom is None:
            self._bloom = Bloom(self.ctx)
        occluders = [(o, m) for o, m in visible if not o.get("bloom")]
        self._bloom.render(
            self._screen, glowing, lambda fbo: self._render_early_z(occluders, view_proj, target=fbo),
            view_proj, self._get_model_matrix, self.bloom_intensity)
        self._screen.use()

    def _render_scene(self, camera, pbr_view_proj):
        self._screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        # pbr_program was bound by _bind_pbr_frame_uniforms; the skeletal and morph programs are separate GL
        # programs with their own uniforms, bound here (morph only if the scene has any morph object).
        skeletal_view_proj = self._bind_frame_environment(self.skeletal_program, camera)
        self._skeletal_view_proj = skeletal_view_proj
        morph_objects = [o for o in self.dynamic_objects if o.get("has_morph")]
        morph_view_proj = self._bind_frame_environment(self.morph_program, camera) if morph_objects else None

        # Frustum culling from this frame's matrix. OPAQUE (back-face culled) and MASK (double-sided, discards
        # below the cutoff) draw in this pass; BLEND objects are collected for _render_transparent_objects, which
        # runs after the skybox. A culled BLEND object never reaches it.
        frustum_planes = extract_frustum_planes(pbr_view_proj)

        blended_objects = []
        opaque_list = []           # visible opaque/MASK objects, drawn below (depth pre-pass, then colour)
        # Whether anything visible uses SSR: render() skips the expensive SSR passes when nothing does (e.g. the
        # camera facing away from the water).
        any_ssr_visible = False
        vis_statics, vis_want_culling, vis_has_lightmap, vis_group = self._visible_statics(frustum_planes)
        tagged_objects = (
            *zip(vis_statics, (False,) * len(vis_statics), vis_want_culling, vis_has_lightmap, vis_group),
            # morph objects have their own program and pass, below
            *((o, True, None, None, None) for o in self.dynamic_objects
              if not o.get("has_morph") and self._is_visible(o, frustum_planes, movable=True)),
        )
        # Parallel to opaque_list: (want_culling, has_lightmap, material_group) precomputed for a static object
        # (_static_render_arrays), Nones for a dynamic one (read live - there are few).
        opaque_render_flags = []
        last_group = None          # material_group of the previous object drawn (see below)
        lights_bound_at = None     # where the lights/probe uniforms were last bound for
        lights_bound = None        # ...and which (lights, probe) pair, for objects sharing a cell
        for obj, movable, want_culling, has_lightmap, group in tagged_objects:
            if obj.get("reflection_mode") == "ssr":
                any_ssr_visible = True
            if obj.get("alpha_mode") == "BLEND":
                blended_objects.append((obj, movable))   # movable: needed for its depth sort
                continue
            opaque_list.append((obj, movable))
            opaque_render_flags.append((want_culling, has_lightmap, group))

        self._frame_opaque_visible = opaque_list       # reused by the SSR depth pre-pass
        early_z = EARLY_Z_PREPASS and bool(opaque_list)
        if early_z:
            self._render_early_z(opaque_list, pbr_view_proj)
            # The colour pass now only needs to match the depth just written: equal-or-nearer
            # passes, and nothing more is written.
            self.ctx.depth_func = "<="
            self._screen.depth_mask = False

        # Face culling is set to back-face at the top of this method; it's only re-set below when
        # an object needs it different (None = unknown, after the early-Z pass left its own state).
        culling_on = None if early_z else True
        for (obj, movable), (want_culling, has_lightmap, group) in zip(opaque_list, opaque_render_flags):
            # No culling for MASK or a double_sided OPAQUE material (a water plane seen from both sides).
            is_static = want_culling is not None   # precomputed flags only exist for statics
            if not is_static:
                want_culling = not (obj.get("alpha_mode") == "MASK" or obj.get("double_sided"))
            if want_culling != culling_on:
                if want_culling:
                    self.ctx.enable(moderngl.CULL_FACE)
                    self.ctx.cull_face = "back"
                else:
                    self.ctx.disable(moderngl.CULL_FACE)
                culling_on = want_culling
            # A lightmapped object never reads the point lights; a mover (or an unbaked static) gets the ones
            # nearest to it. Uniforms persist, so a second material of the same mover needs no rebind.
            if movable or not (has_lightmap if is_static else obj.get("lightmap_texture")):
                if lights_bound_at is None or obj["position"] != lights_bound_at:
                    lit = self._mover_lighting(obj)
                    if lit is not lights_bound:
                        self._bind_lighting(self.pbr_program, lit)
                        lights_bound = lit
                    lights_bound_at = obj["position"]
            model_matrix = self._get_model_matrix(obj)
            # Objects tagged with the same material_group (a burst of gibs) have identical
            # textures and material buffers: after the first, only the transform changes.
            if not is_static:
                group = obj.get("material_group")
            if group is not None and group == last_group:
                bind_transform_only(self.pbr_program, model_matrix, pbr_view_proj)
            else:
                bind_material(self.pbr_program, obj, model_matrix, pbr_view_proj)
            last_group = group
            obj["vao"].render()

        if early_z:
            self.ctx.depth_func = "<"
            self._screen.depth_mask = True

        # Skeletal objects all draw opaque and back-face culled (their loader has no alpha modes).
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

        skeletal_lit = None
        cam_pos = glm.vec3(camera.position)
        by_material = {}
        seen = {}       # (position object, box) -> (distance, outside): a character and its gun share one
        for obj in self.skeletal_objects:
            if not obj.get("visible_in_color", True):
                continue
            if obj.get("frustum_cull") is not None:
                # A character: remember how far it is (animation, shadow, reflection and lighting
                # detail all scale with that - see Scene.update and the passes below) and skip it
                # when it's out of view or past its draw distance.
                key = (id(obj["position"]), obj["frustum_cull"])
                found = seen.get(key)
                if found is None:
                    found = seen[key] = (glm.distance(glm.vec3(obj["position"]), cam_pos),
                                         self._skeletal_culled(obj, frustum_planes))
                dist, outside = found
                obj["_view_dist"] = dist
                limit = obj.get("max_draw_distance")
                obj["_culled"] = outside
                if outside or (limit is not None and dist > limit):
                    continue
            by_material.setdefault(self._skeletal_signature(obj), []).append(obj)

        # Drawn grouped by material so consecutive characters (ten players, ten guns) with the
        # same textures and material buffer skip re-binding them - only the transform and the
        # bones change from one to the next.
        bound_signature = None
        for signature, objects in by_material.items():
            for obj in objects:
                model_matrix = self._get_model_matrix(obj)

                # Never lightmapped: always its own nearest lights.
                lit = self._mover_lighting(
                    obj, self._SKELETAL_LIGHT_CELL if obj.get("_view_dist", 0.0) < self._FAR_CHARACTER else 4.0)
                if lit is not skeletal_lit:
                    self._bind_lighting(self.skeletal_program, lit)
                    skeletal_lit = lit
                # bind_material works for any program with pbr_shader's fragment uniforms (skeletal, morph).
                if signature == bound_signature:
                    bind_transform_only(self.skeletal_program, model_matrix, skeletal_view_proj)
                else:
                    bind_material(self.skeletal_program, obj, model_matrix, skeletal_view_proj)
                    bound_signature = signature
                bind_bone_matrices(obj)
                bind_dissolve(self.skeletal_program, obj)   # every object, or one's dissolve leaks onto the next
                obj["vao"].render()
                if self._active_hat(obj) is not None:
                    self._draw_hat(obj)
                    bound_signature = None      # the hat rebinds an untinted material buffer

        self._morph_view_proj = morph_view_proj
        if morph_objects:
            self.ctx.enable(moderngl.CULL_FACE)
            self.ctx.cull_face = "back"
            morph_lit = None
            for obj in morph_objects:
                if not obj.get("visible_in_color", True) or not self._is_visible(obj, frustum_planes, movable=True):
                    continue
                if obj.get("alpha_mode") == "BLEND":
                    blended_objects.append((obj, True))     # drawn after the skybox, see below
                    continue
                model_matrix = self._get_model_matrix(obj)
                lit = self._mover_lighting(obj)
                if lit is not morph_lit:
                    self._bind_lighting(self.morph_program, lit)
                    morph_lit = lit
                bind_material(self.morph_program, obj, model_matrix, morph_view_proj)
                self.morph_program["u_morph_weight"].value = obj.get("morph_weight", 0.0)
                obj["vao"].render()

        return blended_objects, any_ssr_visible

    def _render_transparent_objects(self, camera, blended_objects, pbr_view_proj):
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
        blended_objects = sorted(blended_objects, key=_distance_sq, reverse=True)   # farthest first

        # Depth-tested but not depth-written: otherwise the first blended surface at a pixel hides every one behind
        # it - even other faces of itself (it showed as sky through foliage/water). (moderngl's depth mask lives
        # on the Framebuffer, not the Context.)
        self._screen.depth_mask = False
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # Everything else is bound frame-wide already; only this frame's SSR grab is new (for "ssr" water).
        ssr_color, ssr_depth = self._ssr_textures()
        bind_ssr_textures(self.pbr_program, ssr_color if settings.ssr else None, ssr_depth)
        lights_bound = None
        lights_morph = False
        for obj, _movable in blended_objects:
            lit = self._mover_lighting(obj, per_object=True)   # never lightmapped: always its nearest lights
            morph = obj.get("has_morph")   # a morph object's VAO belongs to morph_program
            prog = self.morph_program if morph else self.pbr_program
            if lit is not lights_bound or morph != lights_morph:
                bind_packed_lighting(prog, lit[2])
                lights_bound = lit
                lights_morph = morph
            model_matrix = self._get_model_matrix(obj)
            bind_material(prog, obj, model_matrix, pbr_view_proj if not morph else self._morph_view_proj)
            if morph:
                prog["u_morph_weight"].value = obj.get("morph_weight", 0.0)
            obj["vao"].render()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"
        self._screen.depth_mask = True

    def _render_viewmodels(self, camera):
        """Draws the first-person viewmodels (see add_viewmodel) last, over
        everything else. They must never poke into a wall they're standing
        right against, and the usual fix (clear the depth buffer first)
        isn't available here: moderngl's Framebuffer.clear always clears
        colour too, and re-filling depth with a fullscreen pass would cost
        a whole extra draw. Instead the projection's depth output is
        squashed into the nearest 5% of the depth range - so a viewmodel
        always passes the depth test against the scene - at zero cost: it's
        just a different matrix. Everything else (lights, shadows, sky
        colours) is already bound frame-wide by _bind_pbr_frame_uniforms/
        _render_scene, so a visible part adds a couple of uniform writes and
        one draw call, and with nothing visible (the third-person/no-arms
        case) this returns before touching any GL state."""
        # hidden_while_scoped only stops the draw: a scoped gun keeps animating (muzzle/scope points stay live).
        visible = [o for o in self.viewmodel_objects
                   if o.get("viewmodel_visible") and not o.get("hidden_while_scoped")]
        if not visible:
            return

        # z_ndc -> z_ndc * s + (s - 1): [-1, 1] maps onto [-1, -1 + 2s].
        squash = glm.mat4(1.0)
        squash[2][2] = _VIEWMODEL_DEPTH_SCALE
        squash[3][2] = _VIEWMODEL_DEPTH_SCALE - 1.0
        # The camera's projection with a much closer near plane (see _VIEWMODEL_NEAR).
        viewmodel_proj = glm.perspective(glm.radians(camera.fov), camera.aspect, _VIEWMODEL_NEAR, camera.far)
        view_proj = squash * viewmodel_proj * camera.get_view_matrix()

        prog = self.skeletal_program
        # The gun ignores shadows cast by movers - above all your own body, which sits right
        # behind it. This program is shared with the characters, so it's reset at the end.
        if _has_uniform(prog, "u_ignore_mover_shadows"):
            prog["u_ignore_mover_shadows"].value = 1
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"
        # Opaque parts first, then alpha-BLEND parts (a gun's glass) over
        # them, blending without writing depth.
        ordered = ([o for o in visible if o.get("alpha_mode") != "BLEND"]
                   + [o for o in visible if o.get("alpha_mode") == "BLEND"])
        blending = False
        lights_bound = None
        for obj in ordered:
            is_blend = obj.get("alpha_mode") == "BLEND"
            if is_blend != blending:
                blending = is_blend
                if blending:
                    self.ctx.enable(moderngl.BLEND)
                    # Set explicitly: the muzzle-flash particles drawn just before leave their own additive blend func
                    # (that broke the Gouda Gun's text transparency while firing).
                    self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    self._screen.depth_mask = False
                else:
                    self.ctx.disable(moderngl.BLEND)
                    self._screen.depth_mask = True
            # All of a weapon's parts sit at the camera: one lookup, at its own position (not the coarse cells,
            # which made the gun's lighting visibly step as you walked).
            if lights_bound is None:
                lights_bound = self._mover_lighting(obj)[2]
                bind_packed_lighting(prog, lights_bound)
            bind_material(prog, obj, self._get_model_matrix(obj), view_proj)
            bind_bone_matrices(obj)
            bind_dissolve(prog, obj)   # resets a dissolve a character drawn earlier left on the shared program
            obj["vao"].render()
        if blending:
            self.ctx.disable(moderngl.BLEND)
            self._screen.depth_mask = True
        if _has_uniform(prog, "u_ignore_mover_shadows"):
            prog["u_ignore_mover_shadows"].value = 0
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = "back"

    # =============================================================
    # RENDER
    # =============================================================

    def render(self, camera, prog=None):
        self._model_cache = {}
        try:
            self._render_frame(camera, prog)
        finally:
            self._model_cache = None

    def _render_frame(self, camera, prog=None):
        target = self._ensure_target()
        if target is not None:
            self._screen_override = target.fbo
            target.begin()
        else:
            self._screen_override = None
        prof = self.profiler if self.profiler is not None and self.profiler.enabled else None
        t = _time.perf_counter() if prof else 0.0
        self._update_attachments()
        self._caster_sphere = ("unbounded",)       # _render_shadows narrows this down
        if prof:
            t = self._lap(prof, "  render: attachments", t)
        if ENABLE_SHADOWS:
            self._render_shadows(camera)
        if prof:
            t = self._lap(prof, "  render: shadows", t)

        pbr_view_proj = self._bind_pbr_frame_uniforms(camera)
        self._bind_caster_sphere()

        if prof:
            t = self._lap(prof, "  render: frame uniforms", t)
        blended_objects, any_ssr_visible = self._render_scene(camera, pbr_view_proj)
        if prof:
            t = self._lap(prof, "  render: main pass", t)

        if self.skybox_textures is not None:
            render_skybox(
                self.ctx, self.skybox_program, self.skybox_vao, self.skybox_textures,
                self.skybox_average_colors, camera, edge_fade=self.skybox_edge_fade
            )

        if self.equirect_skybox_texture is not None:
            render_equirect_skybox(
                self.ctx, self.equirect_skybox_program, self.equirect_skybox_vao,
                self.equirect_skybox_texture, camera, exposure=self.equirect_exposure,
                apply_tonemap=self.equirect_is_hdr, yaw_degrees=self.skybox_yaw,
                fov_degrees=self.skybox_fov, flat_vertical_span=self.skybox_flat_span, fog=self.fog
            )

        # SSR after the skybox (a ray reaching open sky must grab the real sky), and only when something on screen
        # uses it - the grab/pre-pass redraws every opaque object and was a large cost paid regardless.
        if prof:
            t = self._lap(prof, "  render: skybox", t)
        ssr_on = settings.ssr and self._ssr_on
        if any_ssr_visible and ssr_on:
            self._grab_scene_textures(camera)
            if prof:
                t = self._lap(prof, "  render: ssr grab+prepass", t)
            self._render_ssr_pass(camera, pbr_view_proj)
            if prof:
                t = self._lap(prof, "  render: ssr water redraw", t)

        # After the skybox and the SSR redraw (which must be the last thing to touch the water before blending).
        self._render_transparent_objects(camera, blended_objects, pbr_view_proj)
        if prof:
            t = self._lap(prof, "  render: transparent", t)
        self._render_bloom(camera, pbr_view_proj)
        if prof:
            t = self._lap(prof, "  render: bloom", t)
        if self.before_viewmodels is not None:
            self.before_viewmodels()
        self._render_viewmodels(camera)
        if prof:
            t = self._lap(prof, "  render: viewmodels", t)


    # =============================================================
    # DESTROY
    # =============================================================

    def destroy(self):
        # Actors first, before their own objs/body are ripped out from
        # under them by the static_objects/dynamic_objects/physics teardown
        # below - ActorBase.destroy() is what actually calls remove_static_
        # object/remove_dynamic_object/physics.remove_body for each one, so
        # running it after those would double-release (or release nothing,
        # if iterating an already-cleared list).
        for actor in self.actors:
            actor.destroy()
        self.actors = []
        if self.gibs is not None:
            self.gibs.destroy()
            self.gibs = None
        for obj in self.static_objects:
            self._release_object(obj)
        for obj in self.dynamic_objects:
            self._release_object(obj)
        for obj in self.skeletal_objects:
            self._release_skeletal_object(obj)
        # viewmodel_objects is its own separate list (see add_viewmodel),
        # not a subset of skeletal_objects above, but uses the exact same
        # GL resource shape/release helper (remove_viewmodel's own
        # remove_skeletal call already relies on this) - skipping it here
        # would leak every weapon viewmodel's VAO/texture on scene teardown.
        for obj in self.viewmodel_objects:
            self._release_skeletal_object(obj)

        if self._bloom is not None:
            self._bloom.destroy()
            self._bloom = None
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
            "pbr_program", "morph_program", "shadow_program", "morph_shadow_program", "bake_program",
            "dilate_program", "dilate_quad_vao", "dilate_quad_vbo",
            "blur_program", "blur_quad_vao", "blur_quad_vbo",
            "skeletal_program", "skeletal_shadow_program",
            "skybox_program", "skybox_vao", "skybox_vbo",
            "equirect_skybox_program", "equirect_skybox_texture",
            "equirect_skybox_vao", "equirect_skybox_vbo",
            "_ssr_fbo", "_ssr_depth_fbo", "_ssr_color_texture", "_ssr_depth_texture",
            "_static_shadow_texture", "_target",
        )
        for attr in program_and_buffer_attrs:
            _release(getattr(self, attr))
            setattr(self, attr, None)
        self._screen_override = None

        if self.skybox_textures is not None:
            for tex in self.skybox_textures:
                _release(tex)
            self.skybox_textures = None
            self.skybox_average_colors = None

        self.static_objects.clear()
        self.dynamic_objects.clear()
        self.skeletal_objects.clear()
        self.viewmodel_objects.clear()

    def _release_skeletal_object(self, obj):
        for key in ("vao", "shadow_vao", "_render_ibo", "_shadow_ibo", "texture", "bone_ubo", "_material_ubo",
                    "_material_ubo_untinted", "tint_mask_texture"):
            _release(obj.get(key))

        for vbo_dict_key in ("_render_vbos", "_shadow_vbos"):
            for vbo in obj.get(vbo_dict_key, {}).values():
                _release(vbo)

        for hat in obj.get("hats", {}).values():
            for key in ("vao", "shadow_vao", "ibo", "_shadow_ibo", "texture"):
                _release(hat.get(key))
            for vbo_dict_key in ("vbos", "_shadow_vbos"):
                for vbo in hat.get(vbo_dict_key, {}).values():
                    _release(vbo)

    def _release_object(self, obj):
        """Frees an object's GL resources - its VAOs, textures AND vertex/index buffers (moderngl frees nothing on
        its own: gc_mode is off). Releasing twice is a no-op, so prop copies sharing one original are fine."""
        for key in self._OBJECT_GL_KEYS:
            _release(obj.get(key))
        for buffer in obj.get("_buffers", ()):
            _release(buffer)
