"""
Blinn-Phong (Source-engine-style) shading + cascaded-shadow-aware
material binding.

This used to be a Cook-Torrance PBR shader (GGX normal distribution +
geometry attenuation + Fresnel-Schlick microfacet BRDF). Switched to
Blinn-Phong to match how Source's actual material shading works - its
$phong/$phongexponent/$phongboost VMT parameters are Blinn-Phong, not a
physically-based microfacet model. The practical difference: Blinn-Phong
is just pow(dot(N, H), shininess) - a simpler, more artist-tunable
highlight shape, without GGX's distribution curve, without a geometry/
shadowing attenuation term, and without a Fresnel edge-brightening
curve. This is a purely stylistic swap in the GLSL math below - none of
the Python-side binding code changed, since u_metallic/u_roughness are
still uploaded exactly the same way; the shader just derives a Phong
shininess exponent and a specular tint from them now instead of feeding
a GGX/Fresnel pipeline.

Point lights are always unshadowed in real time. Shadow-tested point
light contributions against static geometry are baked once via
lightmap_baker.py's bake_static_lighting(), not recomputed every frame -
see that file's docstring for why (arbitrary "stationary" light counts
without hitting shader register/texture-unit limits).

The directional (sun) light works the same way now: baked into the
lightmap for any object that has one (static geometry), real-time via
CascadedShadowMap only for objects that don't (dynamic/skeletal -
moving objects, whose transform changes every frame so their own
lighting can't be baked). See main()'s own u_has_lightmap branch below
and Scene.bake_static_lighting's docstring - this is what actually
casts/receives a moving object's shadow correctly (static geometry
still renders into the real-time cascades purely as an occluder for
that, even though it no longer samples them for its own shading).

This used to also carry real-time point-light shadow maps (6 depth
textures + a mat4 per shadow-casting point light, sized by a
MAX_SHADOW_POINT_LIGHTS constant). That was removed after it turned out
to be the actual cause of a GLSL "Constant register limit exceeded"
link error once light counts grew - large per-light mat4/sampler arrays
burn through a legacy shader compiler's fixed register budget fast, and
that cost scaled with light count no matter how efficiently the arrays
were packed. Baking sidesteps the problem entirely: a bake shader only
ever needs uniform space for ONE light at a time (see lightmap_baker.py),
so the number of shadow-casting stationary lights is no longer limited
by this constraint at all.

IMPORTANT - array uniform binding: every array uniform here is assigned
as ONE WHOLE ARRAY (prog["name"].value = tuple(...), or
prog["name"].write(bytes)), never via per-index bracket names like
prog["name[3]"]. moderngl typically only registers a single active
uniform per array (usually "name[0]"), so "name[3]" in prog silently
evaluates False and a per-index write just never happens - this was a
real, silently-broken bug here before. Don't reintroduce per-index
bracket names for arrays in this file.

IMPORTANT - keeping Python and GLSL light counts in sync: NUM_CASCADES
and MAX_POINT_LIGHTS are injected directly into the GLSL source via
FRAGMENT_SHADER_HEADER below, rather than being hardcoded a second time
as #define values in the shader text, so the two can't drift apart.

SHADOW_MAP_RESOLUTION below still has to match CascadedShadowMap's
resolution argument by convention (not derived from a shared constant).
Passing resolution in as a uniform instead would remove this last
manual-sync point, if that's ever worth doing.
"""

import struct

import numpy as np

# Fixed texture units used when binding materials.
TEX_UNIT_ALBEDO = 0
TEX_UNIT_METALLIC_ROUGHNESS = 1
TEX_UNIT_NORMAL = 2  # glTF normalTexture - see FRAGMENT_SHADER_BODY's getNormalFromMap
TEX_UNIT_SHADOW_START = 3  # cascades occupy TEX_UNIT_SHADOW_START..+cascade_count-1
MAX_SHADOW_CASCADES = 3

# Total point lights the real-time shader evaluates, all unshadowed.
# Was capped at 4 historically because of a real "Constant register
# limit exceeded" GLSL link error - but that error came from a
# DIFFERENT, since-removed feature (real-time per-light shadow maps: 6
# depth-texture samplers + a mat4 per shadow-casting point light - see
# this module's own docstring above). What THIS constant gates today is
# just three plain vec3/float arrays (u_point_light_pos/color/radius) -
# far cheaper, and no longer anywhere near that old register budget.
# Raised to cover every point light a scene realistically bakes (see
# Scene._nearest_point_lights - a real-time object still only pays for
# however many lights are actually near it, this is just the ceiling),
# confirmed to still compile/link headlessly at this size.
MAX_POINT_LIGHTS = 16

# Free since real-time point-light shadows were removed (they used to
# occupy this range).
TEX_UNIT_LIGHTMAP = TEX_UNIT_SHADOW_START + MAX_SHADOW_CASCADES  # = 6

# A SECOND cascade set, containing ONLY dynamic/skeletal (movable)
# casters - never static geometry (see Scene._render_shadows). This is
# what a lightmapped (static) surface samples to receive a real-time
# shadow cast by, say, the player walking across it, WITHOUT double-
# counting: the main u_shadow_maps cascades above still include static
# casters too (needed so a movable object is correctly shadowed by a
# static wall/roof - see bind_frame_uniforms), but a static/lightmapped
# surface never samples THOSE, since a static object's own shadowing
# from other static geometry is already baked into its lightmap (see
# lightmap_baker.py/pbr_shader.py's u_has_lightmap branch in main()).
# Sampling the full set from a static surface would shadow it a SECOND
# time for exactly the same static occluder it was already baked dark
# under.
TEX_UNIT_MOVABLE_SHADOW_START = TEX_UNIT_LIGHTMAP + 1  # = 7, occupies 7..9

# The scene's own equirectangular skybox, reused as a crude reflection
# source for near-mirror-smooth materials - see FRAGMENT_SHADER_BODY's
# sample_reflection_env/u_reflection_env and bind_reflection_environment.
TEX_UNIT_REFLECTION_ENV = TEX_UNIT_MOVABLE_SHADOW_START + MAX_SHADOW_CASCADES  # = 10

# Screen-Space Reflections: a per-frame grab of the already-rendered
# opaque scene's own color+depth (Scene.enable_screen_space_reflections/
# _grab_scene_textures) - what a material's u_reflection_mode=1 ("ssr")
# ray-marches instead of the plain skybox u_reflection_env above. See
# bind_ssr_textures/FRAGMENT_SHADER_BODY's own ssr_reflect for exactly
# how the two differ.
TEX_UNIT_SSR_COLOR = TEX_UNIT_REFLECTION_ENV + 1  # = 11
TEX_UNIT_SSR_DEPTH = TEX_UNIT_REFLECTION_ENV + 2  # = 12

# A single fixed, whole-level, camera-INDEPENDENT depth map covering
# every static object's own shadow contribution - see Scene._ensure_
# static_shadow_map's own docstring for why this exists as a separate
# thing from u_shadow_maps/u_movable_shadow_maps above: those are both
# tightly re-fit to the camera's current view frustum every frame
# (CascadedShadowMap.update), so they can never be cached across frames
# without going stale the instant the camera moves - static geometry
# itself never moves though, so building ITS shadow contribution once,
# camera-independent, and just re-sampling that every frame (see
# calculate_static_shadow) replaces what used to be a full re-render of
# every static object into every cascade, every single frame.
TEX_UNIT_STATIC_SHADOW = TEX_UNIT_SSR_DEPTH + 1  # = 13
TEX_UNIT_TINT_MASK = TEX_UNIT_STATIC_SHADOW + 1  # = 14 - player-color mask, see u_tint_mask_texture

# A static object's SUN-ONLY baked lightmap (see Scene.bake_static_
# lighting's own sun_lightmap_texture comment and calculate_shadow's
# call site in main() below) - same resolution/format/UVs as u_lightmap,
# baked from ONLY bake_directional_light (never additively combined with
# any point light), so a moving object's own shadow (which only ever
# blocks the SUN) can subtract exactly its own known contribution back
# out of u_lightmap's combined sun+point total, instead of darkening the
# point-light portion too.
TEX_UNIT_SUN_LIGHTMAP = TEX_UNIT_STATIC_SHADOW + 1  # = 14

# Uniform-buffer binding point for MaterialBlock (see bind_material's
# own docstring for why this replaced 9 individual per-object uniform
# writes) - distinct from skeletal_shader.py's BONE_UBO_BINDING (0) so
# a skeletal draw, which needs both blocks bound at once, doesn't have
# one silently overwrite the other.
MATERIAL_UBO_BINDING = 1

# Matches the fragment shader's own u_alpha_mode int encoding exactly
# (see FRAGMENT_SHADER_BODY's own comment on that uniform) - glTF's
# alphaMode string (model_loader.py's _extract_material/Scene._load_
# object's "alpha_mode" field) mapped to the int this shader actually
# switches on. Anything not in this dict (there isn't one - every
# _extract_material call resolves to one of these 3, but bind_material
# still falls back to 0/OPAQUE defensively) reads as OPAQUE.
_ALPHA_MODE_TO_INT = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}

# Injected directly into the GLSL #defines below so Python and GLSL
# can never drift apart the way NUM_CASCADES historically could.
FRAGMENT_SHADER_HEADER = f"""
#version 330

#define NUM_CASCADES {MAX_SHADOW_CASCADES}
#define SHADOW_MAP_RESOLUTION 2048.0
#define MAX_POINT_LIGHTS {MAX_POINT_LIGHTS}
"""

VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;
in vec2 in_lightmap_uv;
// xyz = tangent, w = handedness (+-1) - see model_loader.py's
// _compute_tangents. Always present on any VAO built for THIS program
// (the shadow/bake programs use their own separate, unrelated vertex
// shaders - see scene_base.py - so they never need this at all), even
// for a material with no normal map, since whether any GIVEN material
// happens to use one isn't known until after this VAO is already built.
in vec4 in_tangent;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out vec2 v_lightmap_uv;
out vec4 v_tangent;

void main() {
    v_position = (u_model * vec4(in_position, 1.0)).xyz;
    mat3 normal_matrix = mat3(transpose(inverse(u_model)));
    v_normal = normal_matrix * in_normal;
    v_color = in_color;
    v_uv = in_uv;
    v_lightmap_uv = in_lightmap_uv;
    // Handedness (w) is a plain sign flag, not a direction - carried
    // through unchanged, only the xyz direction needs the same normal-
    // matrix transform v_normal itself gets (so tangent and normal stay
    // consistent under non-uniform scale/rotation).
    v_tangent = vec4(normal_matrix * in_tangent.xyz, in_tangent.w);
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER_BODY = """
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform vec3 u_eye_pos;
uniform mat4 u_view_matrix;

// Every one of these 9 factors is FIXED per object (read once from
// the glTF material at load time - see model_loader.py's _extract_
// material) - none of them ever change frame to frame the way u_mvp/
// u_model do. They used to be 9 separate uniforms, individually
// rebuilt into a Python dict and rewritten via 9 separate prog[name]=
// calls in bind_material() EVERY OBJECT, EVERY FRAME - confirmed via
// CPU profiling as a real, avoidable cost once per-object draw counts
// grew into the hundreds. Packed into one uniform buffer instead,
// uploaded ONCE per object (cached on the object itself, see bind_
// material) and merely re-bound (a single cheap call, no data upload)
// on every subsequent draw - the same fix already applied to skeletal
// bone matrices in skeletal_shader.py, for the same reason.
//
// vec4 instead of vec3 for u_emissive specifically to keep every
// member of this block a clean 4/16-byte multiple - std140 packs a
// bare vec3 with 16-byte alignment but only a 12-byte extent, which
// invites an off-by-one padding mismatch between this declaration and
// the Python-side struct.pack call that fills it; an unused .w on a
// vec4 costs 4 bytes to sidestep that class of bug entirely.
layout(std140) uniform MaterialBlock {
    vec4 u_emissive_packed;  // .xyz = emissive, .w unused
    float u_metallic;
    float u_roughness;
    float u_specular_strength;
    float u_base_alpha;
    int u_has_texture;
    int u_has_metallic_roughness_texture;
    // alpha_mode: 0 = OPAQUE (ignore alpha entirely, always output 1.0
    // - this project's original behavior, still every material's
    // default), 1 = MASK (binary cutout - discard below u_alpha_cutoff,
    // otherwise still fully opaque), 2 = BLEND (real alpha blending -
    // see Scene._render_scene's own separate blended-objects pass,
    // which is what actually enables GL_BLEND; without that this
    // uniform alone wouldn't do anything visually).
    int u_alpha_mode;
    float u_alpha_cutoff;
    int u_has_normal_texture;
    // glTF normalTexture.scale - multiplies the tangent-space normal's
    // XY (the "how bumpy" axes) before renormalizing, per spec; 1.0 is
    // "as authored", not a fixed constant, so materials with a subtle
    // vs. strong normal map (this project's water ripple normal map
    // included, authored at 0.3) actually differ visually.
    float u_normal_scale;
    // Source-style water: TWO copies of the SAME normal map, scrolled
    // (panned) at different speeds/directions and (optionally) tiled at
    // a different scale, then blended - the classic way a repeating
    // tile texture avoids reading as an obviously-repeating grid, and
    // gives that "never quite settles" rippling look real water has.
    // Units are UV-space-per-second; e.g. (0.05, 0.02) scrolls slowly
    // diagonally. (0,0) on layer 2 with uv2_scale=1.0 makes it an exact
    // copy of layer 1 (still panning identically) - effectively a
    // single-layer pan, for a material that doesn't want the dual-layer
    // look. See main()'s own apply_normal_map/u_time (bound once per
    // frame, not per-object - see bind_frame_uniforms) for how these
    // become an actual per-frame UV offset.
    vec2 u_normal_pan1_speed;
    vec2 u_normal_pan2_speed;
    // UV tiling multiplier for the SECOND layer only (relative to the
    // mesh's own authored v_uv - layer 1 always samples v_uv as-is) -
    // 1.0 is the same tiling as layer 1 (only the pan direction/speed
    // differs); a different value (e.g. 2.3 - deliberately non-integer,
    // so the two layers' repeats don't line back up on a simple cycle)
    // further breaks up the repeating-tile look.
    float u_normal_uv2_scale;
    // 0 = "cheap" - this material's env_reflection block (see main())
    // samples the scene's plain skybox texture only, same as before
    // this feature existed. 1 = "ssr" - ray-marches the actual rendered
    // scene instead (Screen-Space Reflections - see ssr_reflect/
    // u_scene_color/u_scene_depth/u_has_scene_grab, and Scene.enable_
    // screen_space_reflections/_render_ssr_pass for how the grab this
    // needs gets produced). Meaningless (silently treated as 0) on any
    // material this scene never actually set up SSR for at all - see
    // u_has_scene_grab's own comment.
    int u_reflection_mode;
    // Base UV tiling multiplier for BOTH normal map layers, applied
    // BEFORE u_normal_uv2_scale/the pan offsets (see apply_normal_map) -
    // multiplies the mesh's own authored v_uv, same convention as
    // u_normal_uv2_scale's own (a value > 1.0 tiles the texture MORE
    // times across the surface, so the ripple pattern reads SMALLER/
    // denser; < 1.0 tiles it fewer times, so the pattern reads BIGGER/
    // more spread out - the inverse of what "scale" might suggest at a
    // glance, since this scales UV coordinates, not the visual pattern
    // size directly). 1.0 (the default) matches this material's
    // behavior before this control existed - the mesh's own authored
    // UV density, unmodified.
    float u_normal_uv_scale;
    // Unused - keeps this block's total size a multiple of 16 bytes
    // (std140 - see the ORIGINAL _pad_material's own comment for the
    // full reasoning, now satisfied by these 3 floats instead since
    // u_normal_uv_scale used up that original slot).
    float _pad_material2a;
    float _pad_material2b;
    float _pad_material2c;
    // Player-color tint (see Scene.set_skeletal_tint): u_tint_color is the
    // chosen sRGB color, u_has_tint gates the whole effect. The mask
    // texture (u_tint_mask_texture) carries the tintable region in its
    // ALPHA and a desaturated version of the base color in its RGB.
    vec3 u_tint_color;
    int u_has_tint;
};

uniform sampler2D u_texture;
uniform sampler2D u_metallic_roughness_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_tint_mask_texture;

// Dynamic/skeletal casters ONLY, tightly re-fit to the camera's own
// view frustum every frame - see Scene._render_shadows' own comment on
// why static objects were dropped from this cascade set (they used to
// be redrawn into it every single frame despite never moving - now
// their shadow contribution instead comes from the separate, fixed,
// camera-independent u_static_shadow_map below, combined together in
// calculate_shadow).
uniform sampler2D u_shadow_maps[NUM_CASCADES];
uniform mat4 u_light_mvps[NUM_CASCADES];
uniform float u_cascade_splits[NUM_CASCADES];
uniform int u_has_shadows;

// calculate_movable_shadow (used by a static/lightmapped surface to
// receive a moving object's own real-time shadow without double-
// counting anything static-on-static already baked into its lightmap)
// used to read a SEPARATE u_movable_shadow_maps/u_movable_light_mvps/
// u_movable_cascade_splits/u_has_movable_shadows uniform set here - a
// second full cascade binding Python had to redo every single frame,
// identical in content to u_shadow_maps above ever since both got
// consolidated onto one CascadedShadowMap instance (see Scene.
// shadow_manager's own __init__ comment). calculate_movable_shadow now
// just reads u_shadow_maps/u_has_shadows directly (see calculate_
// cascade_shadow) - removed entirely rather than left unused, so
// there's no separate binding call left to redundantly pay for.

// Static casters ONLY, fixed/camera-independent (see TEX_UNIT_STATIC_
// SHADOW's own comment) - a single texture/MVP, not a cascade array,
// since a whole static level fits in one fixed ortho volume built once
// rather than needing multiple camera-relative near/far splits.
uniform sampler2D u_static_shadow_map;
uniform mat4 u_static_shadow_mvp;
uniform float u_static_shadow_texel;
uniform int u_has_static_shadow_map;

uniform int u_num_point_lights;
uniform vec3 u_point_light_pos[MAX_POINT_LIGHTS];
uniform vec3 u_point_light_color[MAX_POINT_LIGHTS];
uniform float u_point_light_radius[MAX_POINT_LIGHTS];

// Combined, precomputed diffuse irradiance from EVERY point light in
// the scene (not just the MAX_POINT_LIGHTS nearest ones above),
// trilinearly interpolated from Scene.light_probe_grid at this specific
// object's current position - see light_probes.py's own module
// docstring and calculate_point_light_specular's comment. Bound per-
// object (bind_probe_irradiance), not once per frame - unlike the sun's
// u_light_color/u_light_intensity, this genuinely varies by world
// position, which is different for every real-time-lit object drawn
// this frame. (0,0,0) is a safe default (no point lights / no probe
// grid yet) - matches point_light_sum's own zero-initialized default.
uniform vec3 u_probe_irradiance;

uniform sampler2D u_lightmap;
uniform int u_has_lightmap;
// See TEX_UNIT_SUN_LIGHTMAP's own comment - only meaningful (and only
// bound) when u_has_lightmap is also 1; no separate u_has_sun_lightmap
// flag needed since the two textures are always baked/bound together
// (see Scene.bake_static_lighting and _bind_lightmap).
uniform sampler2D u_sun_lightmap;

// Hemisphere ("skylight"-style) ambient - see bind_environment() and
// Scene.add_equirect_skybox/add_skybox for where these come from.
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;

// The scene's own equirectangular skybox texture (Scene.
// add_equirect_skybox), reused as a crude environment reflection source
// for near-mirror-smooth materials (see sample_reflection_env/main()'s
// own env_reflection block) - see bind_reflection_environment's own
// docstring for why this needs its own separate exposure/is_hdr flags
// rather than reusing render_equirect_skybox's own Python-side
// arguments directly. u_has_reflection_env is 0 whenever the current
// scene never called add_equirect_skybox at all (Scene.add_skybox's
// OLDER cubemap-face skybox has no equivalent single 2D texture to
// sample this way, so it isn't supported as a reflection source).
uniform sampler2D u_reflection_env;
uniform float u_reflection_exposure;
uniform int u_reflection_is_hdr;
uniform int u_has_reflection_env;

// Screen-Space Reflections (SSR) - a per-frame "grab" of the already-
// rendered opaque scene's own color+depth (Scene._grab_scene_textures/
// enable_screen_space_reflections), ray-marched by ssr_reflect for a
// material with u_reflection_mode=1 ("ssr" - see MaterialBlock's own
// comment) instead of the plain skybox u_reflection_env above. u_has_
// scene_grab is 0 whenever the current scene never called enable_
// screen_space_reflections at all, OR during the FIRST (main) pass a
// frame draws before any grab of THIS frame exists yet (see Scene.
// _render_scene's own comment) - either way, every material's env_
// reflection block falls back to the cheap/skybox path instead, so
// "ssr" without a grab to actually sample never reads as black.
uniform sampler2D u_scene_color;
uniform sampler2D u_scene_depth;
uniform int u_has_scene_grab;
// This draw's own camera projection matrix (bound once per frame, same
// camera bind_frame_uniforms already builds u_mvp/u_view_matrix from) -
// ssr_reflect needs this twice: to project each traced view-space ray
// sample into screen space to look up u_scene_depth, and (paired with
// u_near/u_far below) to convert what THAT lookup returns back into a
// view-space depth it can actually compare its own ray against -
// hardware depth is stored non-linearly (denser near the camera), so a
// raw texture(u_scene_depth,...) value isn't directly comparable to a
// view-space Z without this reconstruction.
uniform mat4 u_proj_matrix;
uniform float u_near;
uniform float u_far;
// Seconds, monotonically increasing (Scene's own running clock, see
// bind_frame_uniforms) - the ONLY thing that makes u_normal_pan1_speed/
// u_normal_pan2_speed (both otherwise-static, load-time MaterialBlock
// factors - see its own comment) actually animate: the shader computes
// each layer's ACTUAL per-frame UV offset as pan_speed * u_time itself
// (see apply_normal_map), rather than this being precomputed and
// re-uploaded into the UBO every frame, which would defeat _get_
// material_ubo's whole "build once, just re-bind after that" point.
uniform float u_time;
// A plain incrementing frame counter (wrapped - see Scene.update's own
// comment), used ONLY to vary ssr_reflect's dither pattern frame to
// frame - see its own u_frame_offset comment for why. Not the same
// thing as u_time (seconds) - this needs to change by a large,
// decorrelated-looking amount every single frame regardless of how much
// real time actually elapsed, which a small per-second value wouldn't
// reliably do at a high framerate (consecutive frames could round to
// the same IGN input, defeating the point).
uniform float u_frame_offset;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in vec2 v_lightmap_uv;
// xyz = tangent, w = handedness - see VERTEX_SHADER's own comment. Only
// pbr_shader.py's own VERTEX_SHADER writes REAL per-vertex tangent data
// here; skeletal_shader.py's separate vertex shader (which compiles
// this same fragment shader body - see its own module docstring) writes
// a fixed placeholder instead (see SKELETAL_VERTEX_BODY's own comment)
// since no skeletal material currently sets has_normal_texture=1 to
// ever actually read this - a fragment shader "in" needs SOME matching
// vertex "out" to link at all, regardless of whether that data is
// meaningful for every vertex shader that pairs with it.
in vec4 v_tangent;

out vec4 fragColor;
const float PI = 3.14159265359;

int get_cascade_index(float view_depth) {
    if (view_depth < u_cascade_splits[0]) return 0;
    if (view_depth < u_cascade_splits[1]) return 1;
    // Redundant in terms of actual branching (both arms return 2
    // either way) - deliberately still a REAL read of u_cascade_
    // splits[2], not dead code removed for tidiness. NUM_CASCADES is
    // 3, but u_cascade_splits[2] itself was never actually referenced
    // anywhere before this - both prior comparisons only ever touch
    // indices 0/1. Some GLSL compilers (confirmed via a real AMD crash
    // report: Intel/NVIDIA's didn't, AMD's did) perform dead-element
    // elimination on an array uniform and report a SMALLER "active
    // size" via introspection than the full declared array - moderngl's
    // own .write() validates the byte length it's given against that
    // introspected size, so writing all 3 floats into a uniform the
    // driver now thinks is only 2 floats raised exactly "invalid
    // uniform size", on AMD only. Referencing every declared index at
    // least once is the standard, portable fix for this whole class of
    // uniform-array-vs-"active size" mismatch.
    if (view_depth < u_cascade_splits[2]) return 2;
    return 2;
}

// Normal-offset bias, not a flat depth-comparison fudge factor: nudges
// the tested point a real world-space distance off the surface along
// its own normal before doing the shadow lookup, rather than tweaking
// the comparison threshold in NDC space. This matters especially once
// face culling is disabled for this pass (e.g. to fix peter-panning or
// thin-geometry light bleed) - without culling, a fragment on one side
// of a mesh can find its own near-coincident backface depth in the
// shadow map, and a flat bias has no real separation to work with,
// producing self-shadowing acne. A world-space offset scales correctly
// regardless of depth or angle, which is also how Unreal and other
// engines support two-sided/uncleared shadow passes without acne - it's
// not that they skip bias, it's that they use a more robust kind of it.
// offset_scale grows per cascade since farther cascades cover more
// world space per shadow-map texel and need a proportionally bigger
// offset to stay ahead of that coarser resolution.
// Static casters have no camera-relative cascade to look up at all
// (see u_static_shadow_map's own comment) - just one fixed texture/MVP,
// so this needs neither a cascade index nor a per-cascade offset_scale
// table. 0.05 world-space normal-offset bias for the same reason
// lightmap_baker.py's own calc_directional_shadow uses the same value
// for ITS single whole-scene ortho map (see that function's own
// comment) - this map's world-space texel size is similarly coarse
// (one fixed volume covering the whole static scene) compared to a
// tightly-fit near cascade, and needs a proportionally bigger offset to
// stay ahead of that.
float calculate_static_shadow(vec3 world_pos, vec3 normal) {
    if (u_has_static_shadow_map == 0) return 0.0;

    vec3 offset_pos = world_pos + normal * 0.05;
    vec4 light_space = u_static_shadow_mvp * vec4(offset_pos, 1.0);
    if (light_space.w <= 0.00001) return 0.0;

    vec3 shadow_coord = (light_space.xyz / light_space.w) * 0.5 + 0.5;
    if (any(lessThan(shadow_coord, vec3(0.0))) || any(greaterThan(shadow_coord, vec3(1.0)))) return 0.0;

    float shadow = 0.0;
    vec2 texel_size = vec2(u_static_shadow_texel);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(shadow_coord.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (shadow_coord.z > texture(u_static_shadow_map, uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

// Factored out of calculate_shadow/calculate_movable_shadow below -
// both need EXACTLY this same camera-fit-cascade query (u_shadow_maps
// only ever holds movable casters now - see that uniform's own
// comment), they just each combine it with something different
// afterward (calculate_shadow also max()s in the fixed static map;
// calculate_movable_shadow deliberately doesn't - see its own comment).
// Used to be two full separate uniform sets (u_shadow_maps/u_light_
// mvps/u_cascade_splits AND a second u_movable_shadow_maps/u_movable_
// light_mvps/u_movable_cascade_splits triplet) bound from Python with
// IDENTICAL data at every call site, back when Scene still had two
// separate CascadedShadowMap instances - once those were consolidated
// into one (see Scene.shadow_manager's own __init__ comment), binding
// the same 3 depth textures and 3 mat4s a second time under different
// uniform names was pure waste, confirmed as a real, disproportionately
// large fixed cost for _render_transparent_objects specifically (a
// pass with only a couple of BLEND objects, where this fixed per-call
// overhead dominates far more than it does in the main scene loop's
// many-objects-per-bind_frame_uniforms-call case).
float calculate_cascade_shadow(vec3 world_pos, float view_depth, vec3 normal) {
    if (u_has_shadows == 0 || view_depth <= 0.0) return 0.0;

    int cascade = get_cascade_index(view_depth);

    float offset_scale[NUM_CASCADES] = float[NUM_CASCADES](0.02, 0.05, 0.1);
    vec3 offset_pos = world_pos + normal * offset_scale[cascade];

    vec4 light_space = u_light_mvps[cascade] * vec4(offset_pos, 1.0);
    if (light_space.w <= 0.00001) return 0.0;

    vec3 shadow_coord = (light_space.xyz / light_space.w) * 0.5 + 0.5;
    if (any(lessThan(shadow_coord, vec3(0.0))) || any(greaterThan(shadow_coord, vec3(1.0)))) return 0.0;

    float shadow = 0.0;
    vec2 texel_size = vec2(1.0 / SHADOW_MAP_RESOLUTION);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(shadow_coord.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (shadow_coord.z > texture(u_shadow_maps[cascade], uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

// Combines the camera-fit cascade query above with the fixed static
// map (calculate_static_shadow) via max - shadowed if EITHER says so.
// Splitting the query this way (instead of one shadow source covering
// everything, the way this function used to work before u_shadow_maps
// dropped static casters) is what lets the expensive part - every
// static object being redrawn into every cascade, every frame -  happen
// only ONCE ever (Scene._ensure_static_shadow_map) instead of every
// single frame regardless of whether the camera or any static geometry
// actually changed.
float calculate_shadow(vec3 world_pos, float view_depth, vec3 normal, vec3 light_dir) {
    return max(
        calculate_cascade_shadow(world_pos, view_depth, normal),
        calculate_static_shadow(world_pos, normal)
    );
}

// Same cascade query as calculate_shadow above, but WITHOUT the static
// map combined in - see TEX_UNIT_MOVABLE_SHADOW_START's own comment for
// why a static/lightmapped surface needs exactly that: its own already-
// baked lightmap already includes static-on-static shadowing, so
// combining the static map in again here would double-shadow it a
// second time on top of that.
float calculate_movable_shadow(vec3 world_pos, float view_depth, vec3 normal, vec3 light_dir) {
    return calculate_cascade_shadow(world_pos, view_depth, normal);
}

// Point lights are always unshadowed in real time - see the module
// docstring for why (shadowed contributions are baked instead).
//
// SPECULAR ONLY - this used to also return a diffuse (albedo * NdotL)
// term, added on top of the same per-light sum this function still
// feeds. That diffuse term is now u_probe_irradiance instead (see
// main()'s own u_has_lightmap==0 branch and light_probes.py's module
// docstring) - a single value, precomputed and static-shadow-tested
// against EVERY point light in the scene, not just the MAX_POINT_LIGHTS
// nearest ones this function is still capped to. Keeping a real diffuse
// term here IN ADDITION to the probe's would double-count exactly the
// nearest few lights' contribution (counted once by the probe, a
// second time by this loop), so it was removed rather than kept
// redundant - what's left here is purely the live specular highlight
// (a genuinely view-dependent term the probe's single interpolated
// value can't represent), gated by NdotL so it doesn't light backfacing
// spots. Blinn-Phong: pow(NdotH, shininess) tinted by specular_color.
vec3 calculate_point_light_specular(int i, vec3 N, vec3 V, float shininess, vec3 specular_color, vec3 world_pos) {
    vec3 light_vec = u_point_light_pos[i] - world_pos;
    float dist = length(light_vec);
    vec3 L = light_vec / max(dist, 0.0001);
    vec3 H = normalize(V + L);

    float radius = max(u_point_light_radius[i], 0.01);
    float falloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
    float atten = (falloff * falloff) / (dist * dist + 1.0);

    float NdotL = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);

    vec3 specular = specular_color * spec * NdotL * u_specular_strength;

    return specular * u_point_light_color[i] * atten;
}

// Perturbs a geometric normal with the material's own tangent-space
// normal map (if any) - a no-op (returns N unchanged) when u_has_
// normal_texture is 0, so every other material's lighting is completely
// unaffected. Builds the TBN basis from v_tangent - genuinely per-
// VERTEX data (model_loader.py's _compute_tangents), smoothly
// interpolated across a triangle by the rasterizer exactly the way
// v_normal already is - NOT a per-fragment screen-space-derivative
// reconstruction (dFdx/dFdy of position/UV), which was tried first here
// but produces a piecewise-CONSTANT basis per triangle (a derivative is
// constant across a flat-interpolated triangle), visibly discontinuous
// at every triangle edge where neighboring triangles don't happen to
// share an identical derivative - confirmed as the actual cause of
// visible seam lines across the water surface (worst along the internal
// diagonal of every quad and every shared edge between adjacent quads).
// A smoothly-interpolated per-vertex tangent has no such discontinuity,
// same reason vertex normals are smoothed instead of using each
// triangle's own flat face normal.
vec3 apply_normal_map(vec3 N) {
    if (u_has_normal_texture == 0) {
        return N;
    }

    // Source-style dual-layer scrolling water: two samples of the SAME
    // normal map, panned at independent speeds/directions (see
    // MaterialBlock's own u_normal_pan1_speed/u_normal_pan2_speed
    // comment) and the second additionally re-tiled by u_normal_uv2_
    // scale, then averaged and renormalized - a simple, cheap way to
    // combine two ripple patterns into something that never quite
    // repeats, rather than one obviously-scrolling tile. Both speeds
    // default to (0,0) (see _pack_material_ubo) and uv2_scale to 1.0,
    // which makes this collapse to a single static sample - i.e. every
    // material before this feature existed renders bit-for-bit
    // identically, since pan_speed*u_time is then always exactly zero.
    // u_normal_uv_scale re-tiles BOTH layers before anything else (the
    // mesh's own authored UV density otherwise, at 1.0 - see
    // MaterialBlock's own comment) - applied first so u_normal_uv2_
    // scale's own relative retiling of layer 2 still means what its
    // name says (relative to layer 1's tiling, not the raw mesh UVs).
    vec2 base_uv = v_uv * u_normal_uv_scale;
    vec2 uv1 = base_uv + u_normal_pan1_speed * u_time;
    vec2 uv2 = base_uv * u_normal_uv2_scale + u_normal_pan2_speed * u_time;
    vec3 sample1 = texture(u_normal_texture, uv1).xyz * 2.0 - 1.0;
    vec3 sample2 = texture(u_normal_texture, uv2).xyz * 2.0 - 1.0;
    vec3 tangent_normal = normalize(sample1 + sample2);
    // glTF normalTexture.scale - see MaterialBlock's own u_normal_scale
    // comment. Renormalize after scaling XY, since that changes the
    // vector's length and the TBN transform below needs a unit input.
    tangent_normal.xy *= u_normal_scale;
    tangent_normal = normalize(tangent_normal);

    // Gram-Schmidt re-orthogonalize against N - interpolating v_tangent
    // linearly across a triangle (like any other varying) doesn't
    // guarantee it stays exactly perpendicular to the ALSO-interpolated
    // v_normal at every point inside the triangle, only at its 3
    // vertices where both were originally computed together.
    vec3 T = normalize(v_tangent.xyz - N * dot(N, v_tangent.xyz));
    // v_tangent.w carries handedness (+-1, see model_loader.py's
    // _compute_tangents) rather than cross(N,T) alone, which would get
    // this backwards on a mirrored UV island.
    vec3 B = cross(N, T) * v_tangent.w;
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangent_normal);
}

// Samples the scene's own equirectangular skybox texture as a crude
// "environment reflection" for a mirror-like material (see main()'s own
// env_reflection block) - same longitude/latitude UV mapping skybox.
// py's render_equirect_skybox uses for the visible backdrop itself (see
// its own comment for the V-flip reasoning), so a reflection vector
// pointing at, say, the sun/horizon samples the exact same pixel the
// backdrop shows there. Returns LINEAR radiance, ready to be added
// straight into this shader's own lighting sum alongside direct_light/
// point_light_sum/ambient - NOT the skybox's own display-ready
// tonemapped+gamma output (that pass is the FINAL thing on screen for
// ITS OWN pixels; this sample instead feeds into more lighting math
// first, so it must still be linear here, then goes through this
// shader's single Reinhard+gamma pass at the very end of main(), same
// as every other contribution).
vec3 sample_reflection_env(vec3 dir) {
    float u = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    float v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    vec3 raw = texture(u_reflection_env, vec2(u, v)).rgb * u_reflection_exposure;
    // HDR (.exr) sources are already linear radiance. LDR (.png/.jpg)
    // sources are sRGB-encoded DISPLAY data instead, same as a base
    // color texture, so they need the identical un-gamma this shader
    // already applies to albedo (see main()'s raw_albedo -> albedo)
    // before being combined with other genuinely-linear contributions.
    return u_reflection_is_hdr == 1 ? raw : pow(max(raw, vec3(0.0)), vec3(2.2));
}

// Screen-Space Reflections: ray-marches u_scene_depth in VIEW SPACE
// from this fragment along the reflected view direction, looking for
// where the ray's own depth first goes "behind" (further than) the
// already-rendered scene's depth at that screen position - the
// standard, most common real-time SSR approximation (linear march +
// thickness test + a short binary-search refine to tighten the hit),
// used here instead of a full ray-traced/probe-based approach. Returns
// (color.rgb, weight) - weight 0 means "no hit" (ran off screen,
// exceeded the step budget, or landed within the screen-edge fade
// margin where off-screen data would be needed) so main()'s own env_
// reflection block knows to fall back to the cheap skybox sample
// instead of showing a hard-edged cutoff.
//
// Unlike the mirrored-camera "planar reflection" approach this project
// tried first (removed - naive screen-space sampling of a SEPARATE
// camera's render only lines up for a perfectly flat, screen-aligned
// mirror; it visibly swam/drifted as the real camera moved), this needs
// no second camera or extra full-scene render at all: it traces the
// SAME camera's own already-rendered view, just further along a bounced
// ray, so it's automatically pixel-correct for whatever's actually
// visible on screen and moves exactly in step with the real camera. The
// trade-off (true of every SSR technique, not specific to this
// implementation) is it can only ever reflect what the camera can
// already see - anything off-screen, behind the reflective surface's
// own view, or occluded falls back to the cheap skybox sample instead
// of showing nothing/garbage.
// Interleaved Gradient Noise (Jorge Jimenez, "Next Generation Post
// Processing in Call of Duty: Advanced Warfare") - the standard cheap
// per-pixel dither for exactly this class of ray-march/sampling
// artifact (SSR, SSAO, volumetrics, ...), used here instead of the
// plain sin-hash this file tried first: a sin-based hash has no
// particular spatial structure, so tap-blurring its result (see ssr_
// reflect's own blur) only partially hides the noise, and just scaling
// its amplitude down (also tried) trades banding back in as the
// amplitude drops. IGN's gradient structure is specifically shaped so a
// following blur pass (or, ideally, a temporal accumulation pass this
// project doesn't have yet - see ssr_reflect's own u_frame_offset
// comment for why per-frame variation is used INSTEAD here) averages it
// out far more effectively per unit of blur/accumulation than
// unstructured noise does, at identical cost to compute.
float interleaved_gradient_noise(vec2 pos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pos, magic.xy)));
}

vec4 ssr_reflect(vec3 N, vec3 world_pos) {
    if (u_has_scene_grab == 0) {
        return vec4(0.0);
    }

    vec3 view_pos = (u_view_matrix * vec4(world_pos, 1.0)).xyz;
    vec3 view_normal = normalize(mat3(u_view_matrix) * N);
    vec3 view_dir = normalize(view_pos);
    vec3 reflect_dir = normalize(reflect(view_dir, view_normal));

    const int SSR_MAX_STEPS = 32;
    const int SSR_BINARY_STEPS = 6;
    // Step size scales with distance from the camera - a fixed step
    // would either waste most of its budget over-sampling close up
    // (where screen-space movement per view-space unit is huge anyway)
    // or never reach far enough for anything distant.
    float step_size = max(-view_pos.z, 1.0) * 0.08;
    // Thickness MUST scale with step_size, not be a fixed constant (the
    // actual root cause of persistently visible lines through distant
    // reflected objects, confirmed - NOT primarily a dithering/banding
    // problem, even though the dithering work done earlier still helps
    // with a separate, smaller version of the same symptom): once
    // step_size exceeds a fixed thickness, the ray can step clean PAST
    // a surface in a single stride without ever landing inside the thin
    // acceptance window, so distant geometry is missed systematically,
    // not randomly - falling back to the (differently colored) cheap
    // skybox sample right through the middle of what should be a solid
    // reflected object, which reads as a hard-edged streak/line no
    // amount of start-offset dithering could ever fix, since dithering
    // only changes WHERE in a step the ray starts, not whether the step
    // itself is too coarse to ever catch a hit at all. 1.5x the current
    // step size guarantees the ray can never skip over the window
    // entirely, with a small fixed floor so close-up thickness doesn't
    // shrink to an unusably thin sliver.
    float thickness = max(step_size * 1.5, 0.15);

    // Dither the ray's STARTING offset by a per-pixel fraction of one
    // step - without this, every pixel's march lands on the exact same
    // lockstep sequence of distances from the camera, so two
    // neighboring pixels whose rays are nearly parallel (the common
    // case for a distant reflection, where the whole visible surface
    // subtends a small angle) cross the SAME depth-buffer step boundary
    // at the SAME step index, and hardware depth's own precision loss
    // at distance means a whole band of pixels registers a hit (or
    // doesn't) together - visible as regular banding/lines through
    // distant reflected objects. This is a standard, well-known SSR
    // artifact; dithering the start offset is the standard fix, but a
    // STATIC per-pixel pattern only ever trades the problem, not solves
    // it: turn the dither amplitude up and the regular banding lines
    // break up into a regular (if higher-frequency) noise pattern that
    // still reads as grain; turn it down and the lines come back. Real
    // engines solve this with TEMPORAL accumulation (blend this frame's
    // noisy result with a history buffer reprojected from previous
    // frames, averaging the noise out over time while staying sharp on
    // static content) - out of scope for this project's render pipeline
    // today (no history buffer/reprojection exists anywhere in it yet).
    // u_frame_offset is the lightweight middle ground used instead:
    // varying WHICH pixel position IGN is sampled at, frame to frame
    // (see bind_frame_uniforms - Scene.update increments a plain
    // frame counter), so any residual pattern shifts every frame rather
    // than sitting still as a fixed set of lines - it reads as
    // flickering noise instead, which the eye is far less sensitive to
    // than a static pattern (the same principle full temporal
    // accumulation relies on, just without an actual accumulation
    // buffer smoothing it further).
    float dither = interleaved_gradient_noise(gl_FragCoord.xy + u_frame_offset) * 0.5;

    vec3 ray = view_pos + reflect_dir * step_size * dither;
    vec3 prev_ray = ray;
    bool did_hit = false;

    for (int i = 0; i < SSR_MAX_STEPS; i++) {
        prev_ray = ray;
        ray += reflect_dir * step_size;

        vec4 clip = u_proj_matrix * vec4(ray, 1.0);
        if (clip.w <= 0.0) break;
        vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;

        float scene_ndc_z = texture(u_scene_depth, uv).r * 2.0 - 1.0;
        // Analytic NDC-depth -> view-space-Z reconstruction (the
        // standard closed-form inverse for a symmetric OpenGL
        // perspective projection) - avoids an inverse(u_proj_matrix)
        // every single step, which would be far more expensive than
        // this one expression for identical results.
        float scene_view_z = -(2.0 * u_near * u_far) / (u_far + u_near - scene_ndc_z * (u_far - u_near));

        if (ray.z <= scene_view_z && ray.z > scene_view_z - thickness) {
            did_hit = true;
            break;
        }
    }

    if (!did_hit) {
        return vec4(0.0);
    }

    // Binary search refine between the last two ray positions (one
    // known to be in front of the surface, one known to be behind/at
    // it) - a handful of cheap extra iterations that meaningfully
    // sharpen the hit point past the coarse linear step alone.
    vec3 lo = prev_ray;
    vec3 hi = ray;
    vec2 hit_uv = vec2(0.5);
    for (int i = 0; i < SSR_BINARY_STEPS; i++) {
        vec3 mid = (lo + hi) * 0.5;
        vec4 clip = u_proj_matrix * vec4(mid, 1.0);
        hit_uv = (clip.xy / clip.w) * 0.5 + 0.5;
        float scene_ndc_z = texture(u_scene_depth, hit_uv).r * 2.0 - 1.0;
        float scene_view_z = -(2.0 * u_near * u_far) / (u_far + u_near - scene_ndc_z * (u_far - u_near));
        if (mid.z <= scene_view_z) {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Fade out near the screen edges - a ray that just barely stayed
    // inside [0,1] is one step away from needing off-screen data this
    // technique fundamentally doesn't have, so hard-cutting there would
    // read as a visible rectangular seam instead of a smooth falloff.
    vec2 edge_dist = min(hit_uv, 1.0 - hit_uv);
    float edge_fade = clamp(min(edge_dist.x, edge_dist.y) / 0.1, 0.0, 1.0);

    // A cheap 4-tap box blur around the hit point, not a single sharp
    // sample - the dithering above (needed to avoid banding, see its
    // own comment) means two neighboring fragments' rays can land on
    // slightly different final texels even when reflecting a smooth,
    // uniform surface, which reads as visible per-pixel grain. This
    // isn't a substitute for real temporal accumulation (the standard
    // AAA fix - reusing/blending previous frames' samples via history
    // reprojection - out of scope for the render pipeline here), but a
    // few extra taps meaningfully soften that grain for very little
    // extra cost, and a slightly-blurred reflection is generally more
    // convincing anyway (a perfectly sharp one looks unnaturally crisp
    // next to everything else this shader draws).
    vec2 texel = 1.0 / vec2(textureSize(u_scene_color, 0));
    vec3 color = texture(u_scene_color, hit_uv).rgb;
    color += texture(u_scene_color, hit_uv + vec2(-texel.x, -texel.y)).rgb;
    color += texture(u_scene_color, hit_uv + vec2(texel.x, -texel.y)).rgb;
    color += texture(u_scene_color, hit_uv + vec2(-texel.x, texel.y)).rgb;
    color += texture(u_scene_color, hit_uv + vec2(texel.x, texel.y)).rgb;
    color *= 0.2;

    // Un-gamma - u_scene_color holds DISPLAY-READY data (every other
    // object's OWN tonemap+gamma pass, at the very end of THIS SAME
    // shader, already ran before Scene._grab_scene_textures blit it
    // into u_scene_color earlier this same frame), not linear radiance,
    // unlike sample_reflection_env's own HDR-vs-LDR-aware sky sample
    // (see that function's own docstring for the exact same contract:
    // whatever this returns gets blended into main()'s still-linear
    // `color` BEFORE its own single Reinhard+gamma pass, so returning
    // already-gamma-encoded data here doubles gamma on whatever
    // fraction of it gets mixed in). Confirmed as the actual cause of
    // SSR reflections reading too bright almost regardless of
    // reflectivity/u_specular_strength: gamma is a strongly nonlinear
    // boost (especially through midtones), so even a REDUCED blend
    // weight of an already-brightened value stays disproportionately
    // bright once gamma-encoded a second time - the fix has to be here,
    // at the source, not in how much of it gets mixed in. This can't
    // perfectly undo the Reinhard compression that same earlier pass
    // also applied (tonemapping is lossy - genuine over-1.0 HDR range
    // information is really gone by the time this samples it), but
    // undoing the gamma removes the dominant, doubly-nonlinear half of
    // the mismatch, and matches what sample_reflection_env already does
    // for its own LDR (.png/.jpg) sources.
    color = pow(max(color, vec3(0.0)), vec3(2.2));
    return vec4(color, edge_fade);
}

void main() {
    // Backface-corrected normal - only actually differs from v_normal
    // when face culling is disabled (MASK/BLEND alpha_mode - see
    // Scene._render_scene), where the rasterizer can hand this shader a
    // fragment from a triangle's BACK side. Without flipping here, a
    // back-facing fragment gets lit as if it faced away from every
    // light (NdotL near 0, reading as flat-dark) instead of correctly
    // facing the viewer - gl_FrontFacing is exactly OpenGL's own signal
    // for which side the rasterizer actually produced this fragment
    // from, so this is authoritative regardless of the source mesh's
    // own winding/authoring.
    vec3 N = normalize(v_normal) * (gl_FrontFacing ? 1.0 : -1.0);
    // Applied right after the backface correction above (and before
    // anything below reads N) so every later use of N - NdotL, the
    // specular H-dot, calculate_point_light, calculate_shadow's own
    // normal offset, hemisphere_ambient's N.y - automatically picks up
    // the perturbed normal too, exactly as if the surface's actual
    // geometry were rippled. A no-op when this material has no normal
    // texture (see apply_normal_map's own docstring).
    N = apply_normal_map(N);
    vec3 V = normalize(u_eye_pos - v_position);
    vec3 L = normalize(u_light_dir);
    vec3 H = normalize(V + L);

    vec4 tex_sample = u_has_texture == 1 ? texture(u_texture, v_uv) : vec4(v_color, 1.0);
    vec3 raw_albedo = tex_sample.rgb;
    if (u_has_tint == 1) {
        // Recolor only where the mask's alpha says so (fur - skin, eyes and
        // other detail keep their own color). The mask's RGB is the fur
        // already desaturated, so multiplying by the chosen color keeps the
        // fur's light/dark shading; TINT_BOOST lifts it back up to roughly
        // the original fur brightness since that shading is mid-grey.
        const float TINT_BOOST = 2.2;
        vec4 tint_mask = texture(u_tint_mask_texture, v_uv);
        vec3 tinted = clamp(tint_mask.rgb * u_tint_color * TINT_BOOST, 0.0, 1.0);
        raw_albedo = mix(raw_albedo, tinted, tint_mask.a);
    }
    vec3 albedo = pow(max(raw_albedo, vec3(0.0)), vec3(2.2));

    float alpha = clamp(tex_sample.a * u_base_alpha, 0.0, 1.0);
    if (u_alpha_mode == 1 && alpha < u_alpha_cutoff) {
        discard;
    }

    float metal = clamp(u_metallic, 0.0, 1.0);
    float rough = clamp(u_roughness, 0.04, 1.0);
    if (u_has_metallic_roughness_texture == 1) {
        vec4 mr = texture(u_metallic_roughness_texture, v_uv);
        rough *= mr.g;
        metal *= mr.b;
    }
    rough = clamp(rough, 0.04, 1.0);

    // Repurposing the same metallic/roughness inputs the old PBR path
    // used, but to drive a Phong shininess exponent and specular tint
    // instead of a GGX/Fresnel pipeline - roughly analogous to Source's
    // $phongexponent (tighter highlight = shinier/less rough) and a
    // metal-tinted specular color, without claiming physical accuracy.
    // u_specular_strength is the separate, direct intensity control -
    // matching Source's $phongboost - since roughness/metallic alone
    // only shape the highlight, they don't give independent control
    // over how strong it is.
    float shininess = mix(128.0, 4.0, rough);
    vec3 specular_color = mix(vec3(0.04), albedo, metal);

    float NdotL = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);

    float view_depth = -(u_view_matrix * vec4(v_position, 1.0)).z;

    // The sun's own contribution is only ever computed live here for a
    // MOVING object (no lightmap - dynamic/skeletal, which can't be
    // baked since its transform changes every frame). A lightmapped
    // (static) object gets the sun's diffuse contribution from its
    // lightmap sample below instead - lightmap_baker.py's bake_
    // directional_light bakes it once, including proper static-on-
    // static self-shadowing, into the same texture point lights are
    // baked into (see Scene.bake_static_lighting). Skipping calculate_
    // shadow's 9-tap PCF and the specular term entirely for every
    // lightmapped fragment (rather than computing it and discarding
    // the result) is also the actual point-of-this-change performance
    // win - static surfaces cover most of a typical frame's pixels.
    // calculate_shadow here uses the FULL cascade set (static AND
    // movable casters - see u_shadow_maps), not the movable-only one,
    // so a moving object standing behind/under static geometry is
    // still correctly shadowed by it.
    vec3 direct_light = vec3(0.0);
    if (u_has_lightmap == 0) {
        float shadow_attenuation = 1.0 - calculate_shadow(v_position, view_depth, N, L);

        vec3 diffuse = albedo * NdotL;
        vec3 specular = specular_color * spec * NdotL * u_specular_strength;
        direct_light = (diffuse + specular) * u_light_color * u_light_intensity * shadow_attenuation;
    } else if (rough < 0.15) {
        // A lightmapped (static) surface authored near-mirror-smooth
        // (low roughnessFactor - e.g. this project's water material,
        // roughness=0/metallic=0) still needs a LIVE specular highlight:
        // the sun's glint moves with the view angle, which can't be
        // baked into a static lightmap the way diffuse irradiance can -
        // without this, roughness/metallic are silently meaningless for
        // any lightmapped object, since the u_has_lightmap==1 branch
        // below only ever multiplies the baked texture by albedo, never
        // touching either. Gated behind this roughness check (rather
        // than always paying for it) so the vast majority of ordinary
        // static geometry - walls, floors, rocks, none of which are
        // anywhere near this shiny - keeps the lightmap fast-path's
        // actual point: skipping calculate_shadow's 9-tap PCF entirely.
        // Only genuinely mirror-like statics pay for a live shadow trace
        // here, same cascade set as the non-lightmapped branch above.
        float shadow_attenuation = 1.0 - calculate_shadow(v_position, view_depth, N, L);
        vec3 specular = specular_color * spec * NdotL * u_specular_strength;
        direct_light = specular * u_light_color * u_light_intensity * shadow_attenuation;
    }

    vec3 point_light_sum = vec3(0.0);
    if (u_has_lightmap == 1) {
        // Lightmaps store pure incoming light (irradiance - both the
        // sun's and every baked point light's, additively combined at
        // bake time), same as Source's model - they get MULTIPLIED by
        // the surface's own albedo, not added raw.
        vec3 baked = texture(u_lightmap, v_lightmap_uv).rgb * albedo;

        // A moving object (the player, ...) currently standing between
        // this surface and the sun wasn't there when the bake ran, so
        // it can't be reflected in `baked` above - darken by a REAL-
        // TIME shadow test against the movable-only cascade set
        // (u_movable_shadow_maps, which never contains static
        // geometry - see TEX_UNIT_MOVABLE_SHADOW_START's own comment)
        // to still show a moving object's shadow falling across static
        // ground, without re-darkening for a static occluder that's
        // already baked in.
        // Gate the dynamic caster's darkening by whether the sun could
        // even reach this texel in the first place (calculate_static_
        // shadow - the same single fixed map calculate_shadow above
        // uses). Without this, a moving object standing somewhere the
        // sun is already fully blocked by static geometry (under a
        // roof/canopy) still visibly darkens `baked` here - which is
        // wrong on two counts: baked can include real, unshadowed point-
        // light irradiance at that same texel that has nothing to do
        // with the sun, and even the sun-derived portion of baked is
        // already 0 there, so there is no sunlight left for the moving
        // object to plausibly be blocking. calculate_static_shadow is a
        // single cheap 3x3 tap against a texture already bound every
        // frame regardless (see u_static_shadow_map), so this costs one
        // extra lookup only in the branch that needs it.
        //
        // Which PART of `baked` a moving object's shadow is even allowed
        // to darken matters too, not just whether it's allowed to at
        // all: `baked` is sun+point combined, but a moving object only
        // ever blocks the SUN (it has no real-time point-light shadow of
        // its own - see pbr_shader.py's module docstring). Naively
        // scaling the WHOLE of `baked` down here would also dim any
        // point-light contribution baked into this same texel, even
        // though the moving object standing between here and the sun has
        // nothing to do with whatever point light is also reaching this
        // spot - a floor patch lit by both the sun AND a nearby lamp
        // should stay lamp-lit while a passing character's shadow crosses
        // it, not go dark. u_sun_lightmap holds exactly the sun's own
        // isolated contribution to this same texel (see TEX_UNIT_SUN_
        // LIGHTMAP's own comment) - only that known amount is ever
        // subtracted back out, leaving whatever came from point lights
        // untouched. max(..., 0.0) guards the rare texel where dilation
        // pushed u_sun_lightmap's padding slightly past u_lightmap's own
        // (each dilates from its own, independently-shaped valid region)
        // from going negative.
        // calculate_static_shadow is a SECOND full 9-tap PCF lookup, on
        // top of calculate_movable_shadow's own - only actually needed
        // to compute sun_reaches, which only matters when a moving
        // object is actually casting a shadow on THIS texel
        // (movable_shadow > 0) in the first place: removable is scaled
        // by movable_shadow regardless, so sun_reaches's value is
        // irrelevant whenever movable_shadow is already 0 (the vast
        // majority of lightmapped fragments, most frames - nothing
        // moving is anywhere near most of a static scene's surface).
        // Skipping the second PCF tap entirely in that case (rather
        // than computing it unconditionally and multiplying by zero)
        // is what actually matters here - confirmed via GPU profiling
        // that this branch (taken for most of a frame's pixels - see
        // this whole block's own opening comment) had become the
        // single largest GPU cost in the renderer once this shadow-
        // fill fix added a second PCF tap AND a second texture sample
        // to what used to be lightmapped rendering's whole fast-path
        // point.
        float movable_shadow = calculate_movable_shadow(v_position, view_depth, N, L);
        vec3 removable = vec3(0.0);
        if (movable_shadow > 0.0) {
            float sun_reaches = 1.0 - calculate_static_shadow(v_position, N);
            vec3 baked_sun_only = texture(u_sun_lightmap, v_lightmap_uv).rgb * albedo;
            removable = baked_sun_only * (movable_shadow * sun_reaches);
        }
        point_light_sum = max(baked - removable, vec3(0.0));
    } else {
        // u_probe_irradiance carries the diffuse point-light contribution
        // for this moving object - see calculate_point_light_specular's
        // own comment for why the per-light loop below no longer adds
        // its own diffuse term.
        point_light_sum = albedo * u_probe_irradiance;

        int num_points = min(u_num_point_lights, MAX_POINT_LIGHTS);
        for (int i = 0; i < num_points; i++) {
            point_light_sum += calculate_point_light_specular(i, N, V, shininess, specular_color, v_position);
        }
    }

    // Hemisphere ambient (a simplified, 2-term stand-in for a full
    // spherical-harmonics "skylight"): blends between the sky and
    // ground colors by how much the surface faces up vs down, rather
    // than a single flat ambient constant - an upward-facing floor
    // picks up the sky's color/brightness, a downward-facing ceiling
    // the ground's, and a vertical wall gets an even mix. N.y is in
    // WORLD space here (v_normal is transformed by u_model, not view),
    // so this stays correct regardless of camera orientation.
    vec3 hemisphere_ambient = mix(u_ground_color, u_sky_color, N.y * 0.5 + 0.5);
    vec3 ambient = hemisphere_ambient * albedo;

    // Environment reflection - what actually reads as "shiny/mirror-
    // like" (water, glass, chrome) rather than merely "has a highlight":
    // a single directional light's own specular term above is only ever
    // a tiny, narrow glint (pow(NdotH, shininess) decays to ~0 within a
    // fraction of a degree of the exact reflection angle for a high
    // shininess exponent), so on its own a near-mirror-smooth surface
    // still reads as flat/matte everywhere else on it - exactly the
    // "renders fully rough" symptom this block fixes. Gated the same
    // way the lightmapped-specular branch above is (low roughness only)
    // so ordinary matte/semi-glossy materials are completely unaffected.
    vec3 color = direct_light + point_light_sum + ambient + u_emissive_packed.xyz;
    if (rough < 0.15) {
        // Cheap Schlick Fresnel using the same dielectric/metal F0 this
        // shader's direct specular already uses (specular_color) - not
        // part of the direct-light term itself (this shader is
        // deliberately plain Blinn-Phong there, see the module
        // docstring), but standard and appropriate specifically for an
        // environment-reflection blend, where grazing-angle brightening
        // is exactly what sells "reflective" - real water/glass's
        // signature look, which a flat, angle-independent reflection
        // strength doesn't reproduce.
        float NdotV = max(dot(N, V), 0.0);
        vec3 fresnel = specular_color + (vec3(1.0) - specular_color) * pow(1.0 - NdotV, 5.0);

        // "ssr" (u_reflection_mode==1) ray-marches the real, already-
        // rendered scene instead of the plain skybox - see ssr_reflect's
        // own docstring. ssr_reflect itself already returns weight=0
        // (see its own docstring) whenever there's no grab to sample at
        // all (u_has_scene_grab==0 - this scene never called enable_
        // screen_space_reflections, or this IS the main pass before one
        // exists yet this frame - see Scene._render_scene's own
        // comment) or the ray missed/ran off-screen, so blending its
        // weight against the cheap/skybox sample here makes "ssr"
        // degrade smoothly to "cheap" in every one of those cases
        // rather than ever reading black or hard-cutting at an edge.
        vec3 cheap = u_has_reflection_env == 1 ? sample_reflection_env(reflect(-V, N)) : vec3(0.0);
        vec3 reflected_color = cheap;
        if (u_reflection_mode == 1) {
            vec4 ssr = ssr_reflect(N, v_position);
            reflected_color = mix(cheap, ssr.rgb, ssr.a);
        }

        // Blend the reflection INTO the surface's own look (direct_
        // light/point_light_sum/ambient/emissive, already accumulated
        // into `color` above) rather than piling it on top additively,
        // the way this used to work: an unbounded additive term stacks
        // reflection brightness on top of the water's own diffuse/
        // ambient color with no ceiling, which both overexposes (pushed
        // further into the Reinhard tonemap's compressive range below)
        // and reads as a bright reflection sitting ON the water rather
        // than the water's own surface actually being reflective. The
        // blend weight is still fresnel-driven (grazing angles reflect
        // more, matching real water/glass) but capped well below 1.0 -
        // the water's own base color always shows through by at least
        // 40%, even at the most extreme grazing angle, instead of ever
        // fully whiting out into a flat mirror with no material
        // underneath it at all.
        float reflectivity = clamp(max(fresnel.r, max(fresnel.g, fresnel.b)) * u_specular_strength, 0.0, 0.6);
        color = mix(color, reflected_color, reflectivity);
    }

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    // OPAQUE/MASK always output full alpha (MASK's own transparency is
    // the discard above, a binary cutout - not a blended edge) - only
    // BLEND actually writes a partial alpha, which only visually blends
    // at all because Scene._render_scene's own separate pass for BLEND
    // objects is the one that turns GL_BLEND on to begin with.
    fragColor = vec4(color, u_alpha_mode == 2 ? alpha : 1.0);
}
"""

FRAGMENT_SHADER = FRAGMENT_SHADER_HEADER + FRAGMENT_SHADER_BODY


def bind_material_block(prog):
    """Binds `prog`'s own MaterialBlock uniform block to MATERIAL_UBO_
    BINDING - shared by create_program (pbr) and skeletal_shader.py's
    create_skeletal_program, since both compile FRAGMENT_SHADER_BODY
    (and therefore declare MaterialBlock) verbatim. Called once at
    program creation, not per frame - a program's block-to-binding-point
    mapping doesn't change after that."""
    if _has_uniform(prog, "MaterialBlock"):
        prog["MaterialBlock"].binding = MATERIAL_UBO_BINDING
    return prog


def create_program(ctx):
    return bind_material_block(
        ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    )


def _has_uniform(prog, name):
    """name in prog, but not literally that - moderngl's Program class
    defines __iter__ (yields from its own internal self._members dict)
    and __getitem__, but NO __contains__, so Python's `in` operator
    falls back to the generic __iter__-based membership test: a full
    LINEAR SCAN through every uniform/block this program declares,
    comparing each one against `name`, every single time. Confirmed via
    cProfile as the single largest CPU cost in the entire renderer -
    moderngl's own Program.__iter__ (called wherever `in prog` appears)
    outweighed even _write_uniform's own uniform-setting calls, at
    ~1.5 million calls across a 300-frame profiling run of a ~30-object
    scene. self._members is the exact same dict __iter__ already yields
    from and __getitem__ already indexes into - going through it
    directly here is an ordinary O(1) dict lookup instead, doing exactly
    the same check moderngl's own __contains__ would have done had it
    defined one. Every `name in prog`/`name in some_program` in this
    codebase's per-frame/per-object hot path should go through this
    instead - see this file's own _write_uniform/_bind_material_
    textures/_bind_lightmap/bind_point_lights, scene_base.py's
    _bind_shadow_alpha, and skeletal_shader.py's bind_bone_matrices for
    the actual fixes."""
    return name in prog._members


def _write_uniform(prog, name, value):
    """Set a uniform whether it needs .write() (matrices) or .value = (scalars/vectors)."""
    if not _has_uniform(prog, name):
        return
    if isinstance(value, (bytes, bytearray)):
        prog[name].write(value)
    else:
        prog[name].value = value


def _bind_material_textures(prog, item_data):
    # .filter is a property of the texture itself, set once at creation
    # time (model_loader.py's _upload_texture, skeletal_loader.py's
    # load_texture, and Scene.add_skeletal's texture_path override all
    # do this now) - it never changes for a given texture object, so
    # writing it again here on every single object's every single frame
    # was pure redundant GL state traffic. Confirmed via CPU profiling
    # as part of the same per-object-rebind investigation that led to
    # bind_frame_uniforms.
    for key, unit in (
        ("texture", TEX_UNIT_ALBEDO),
        ("metallic_roughness_texture", TEX_UNIT_METALLIC_ROUGHNESS),
        ("normal_texture", TEX_UNIT_NORMAL),
        ("tint_mask_texture", TEX_UNIT_TINT_MASK),
    ):
        tex = item_data.get(key)
        uniform_name = f"u_{key}"
        if tex and _has_uniform(prog, uniform_name):
            tex.use(location=unit)
            prog[uniform_name].value = unit


def _bind_shadow_uniforms(prog, shadow_manager):
    if shadow_manager is None:
        return

    cascades = shadow_manager.depth_textures[:MAX_SHADOW_CASCADES]
    for i, tex in enumerate(cascades):
        tex.use(location=TEX_UNIT_SHADOW_START + i)

    _write_uniform(prog, "u_has_shadows", 1)
    if _has_uniform(prog, "u_shadow_maps"):
        units = tuple(TEX_UNIT_SHADOW_START + i for i in range(MAX_SHADOW_CASCADES))
        prog["u_shadow_maps"].value = units

    splits = list(shadow_manager.splits)
    while len(splits) < MAX_SHADOW_CASCADES:
        splits.append(shadow_manager.far)
    _write_uniform(
        prog,
        "u_cascade_splits",
        np.array(splits[:MAX_SHADOW_CASCADES], dtype=np.float32).tobytes(),
    )

    _write_uniform(
        prog,
        "u_light_mvps",
        b"".join(m.to_bytes() for m in shadow_manager.light_mvps[:MAX_SHADOW_CASCADES]),
    )


def _bind_lightmap(prog, item_data):
    texture = item_data.get("lightmap_texture")
    if texture is not None and _has_uniform(prog, "u_lightmap"):
        texture.use(location=TEX_UNIT_LIGHTMAP)
        prog["u_lightmap"].value = TEX_UNIT_LIGHTMAP
        _write_uniform(prog, "u_has_lightmap", 1)

        # See TEX_UNIT_SUN_LIGHTMAP's own comment. Always present
        # whenever "lightmap_texture" is (Scene.bake_static_lighting
        # bakes/caches both together for every eligible object - there's
        # no "has a combined lightmap but no sun-only one" state), so
        # this doesn't need its own has-texture guard the way the
        # combined lightmap above does.
        sun_texture = item_data.get("sun_lightmap_texture")
        if sun_texture is not None and _has_uniform(prog, "u_sun_lightmap"):
            sun_texture.use(location=TEX_UNIT_SUN_LIGHTMAP)
            prog["u_sun_lightmap"].value = TEX_UNIT_SUN_LIGHTMAP
    else:
        _write_uniform(prog, "u_has_lightmap", 0)


def _bind_static_shadow_uniforms(prog, texture, light_vp):
    """Binds Scene's own fixed, whole-level static-only shadow map (see
    Scene._ensure_static_shadow_map) - a single texture/MVP, not a
    per-cascade array, since it's one fixed ortho volume built once.
    texture=None (no static geometry yet, or a caller that never wants
    static occlusion at all) clears u_has_static_shadow_map to 0 rather
    than leaving whatever was bound last frame, same reasoning as every
    other optional binding in this function."""
    if texture is None:
        _write_uniform(prog, "u_has_static_shadow_map", 0)
        return
    texture.use(location=TEX_UNIT_STATIC_SHADOW)
    _write_uniform(prog, "u_static_shadow_map", TEX_UNIT_STATIC_SHADOW)
    _write_uniform(prog, "u_static_shadow_mvp", light_vp.to_bytes())
    _write_uniform(prog, "u_static_shadow_texel", 1.0 / texture.size[0])
    _write_uniform(prog, "u_has_static_shadow_map", 1)


def bind_frame_uniforms(
    prog, camera, light_dir, shadow_manager=None,
    light_color=(1.0, 1.0, 1.0), light_intensity=2.0,
    time=0.0, near=None, far=None, frame=0,
    static_shadow_texture=None, static_shadow_light_vp=None,
):
    """Binds everything that's identical for every object drawn with
    `prog` THIS FRAME - camera view/eye, the directional light, and the
    shadow cascades - exactly once, instead of every per-object
    bind_material() call redundantly recomputing/rewriting the same
    values (confirmed via CPU profiling: u_light_mvps alone is a 3-4
    mat4 array rebuilt via Python-side glm.to_bytes()/b"".join() calls
    on every single draw, for data that provably can't have changed
    since the last object). Mirrors the bind_point_lights/
    bind_environment calls already hoisted out of Scene._render_scene's
    per-object loop for the same reason - this closes the one case
    (view/light/shadow uniforms) that pattern hadn't been applied to
    yet.

    Returns view_proj (glm.mat4 = projection * view) so callers can
    build each object's own u_mvp as `view_proj * model_matrix` without
    re-deriving the camera's view/projection matrices per object either
    - call once per program per frame (see Scene._render_scene/
    _render_transparent_objects), not once per object.

    light_color/light_intensity: the directional (sun) light's own
    color and brightness multiplier - see Scene.light_color/
    Scene.light_intensity. Defaults match this shader's previous
    hardcoded behavior (implicitly white at a fixed 2.0 multiplier)
    exactly, for any caller that doesn't pass them.

    A static/lightmapped surface's own real-time shadow from a moving
    object (calculate_movable_shadow) now reads straight off the SAME
    shadow_manager cascades passed above (see calculate_cascade_shadow's
    own comment) - there used to be a separate movable_shadow_manager
    param/second cascade binding here for that, removed once Scene
    stopped keeping two separate CascadedShadowMap instances with
    identical content (see Scene.shadow_manager's own __init__ comment).

    static_shadow_texture/static_shadow_light_vp: Scene's own fixed,
    whole-level static-only shadow map (see Scene._ensure_static_shadow_
    map and calculate_shadow's own comment on why static occlusion comes
    from here now instead of from u_shadow_maps). None/None (the default)
    clears u_has_static_shadow_map, same as every other optional binding
    here - harmless for a scene with no static geometry baked yet.

    time: seconds, monotonically increasing (Scene's own running clock -
    see Scene.update) - drives u_time, the only thing that animates a
    water material's dual-layer normal pan (see MaterialBlock's own
    u_normal_pan1_speed/u_normal_pan2_speed comment). 0.0 default is
    harmless for any material with no pan speed set at all (pan_speed *
    0.0 is always zero regardless).

    near/far: this camera's own near/far clip distances (Camera.near/
    Camera.far) - drives u_near/u_far/u_proj_matrix, used ONLY by an
    "ssr" water material's screen-space ray march (see FRAGMENT_SHADER_
    BODY's own ssr_reflect - it needs these to reconstruct a view-space
    depth from what it samples back out of the scene depth grab). None
    (the default, for a caller that doesn't pass them) leaves u_near/
    u_far unset - harmless for a scene/material that never uses SSR at
    all, since ssr_reflect's own u_has_scene_grab==0 short-circuit means
    they're never actually read in that case either.

    frame: Scene's own plain incrementing frame counter (see Scene.
    update) - drives u_frame_offset, which ONLY exists to vary ssr_
    reflect's per-pixel dither pattern frame to frame (see its own
    comment for why - a lightweight stand-in for real temporal
    accumulation, which this project's render pipeline doesn't have).
    0 (the default) is harmless for a scene/material that never uses
    SSR at all, same reasoning as near/far above."""
    view = camera.get_view_matrix()
    projection = camera.get_projection_matrix()
    view_proj = projection * view

    _write_uniform(prog, "u_view_matrix", view.to_bytes())
    _write_uniform(prog, "u_proj_matrix", projection.to_bytes())
    _write_uniform(prog, "u_light_dir", tuple(light_dir))
    _write_uniform(prog, "u_light_color", tuple(light_color))
    _write_uniform(prog, "u_light_intensity", float(light_intensity))
    _write_uniform(prog, "u_eye_pos", tuple(camera.position))
    _write_uniform(prog, "u_time", float(time))
    # Wrapped, not a raw ever-growing frame count - only the input to a
    # fract()-based noise function (interleaved_gradient_noise), whose
    # behavior only depends on this value modulo something IGN itself
    # already treats periodically; wrapping just keeps the float from
    # growing unbounded over a long session for no benefit.
    _write_uniform(prog, "u_frame_offset", float((frame % 1000) * 37.0))
    if near is not None:
        _write_uniform(prog, "u_near", float(near))
    if far is not None:
        _write_uniform(prog, "u_far", float(far))
    # Explicit default before _bind_shadow_uniforms (which only ever
    # WRITES u_has_shadows=1, never clears it) - this call happens once
    # per frame now rather than once per object, so a shadow_manager
    # that goes from set to None between frames needs this reset here
    # or the shader would keep reading last frame's stale "1" forever.
    _write_uniform(prog, "u_has_shadows", 0)
    _bind_shadow_uniforms(prog, shadow_manager)
    _bind_static_shadow_uniforms(prog, static_shadow_texture, static_shadow_light_vp)

    return view_proj


_REFLECTION_MODE_TO_INT = {"cheap": 0, "ssr": 1}

_MATERIAL_UBO_STRUCT = struct.Struct("<4f4f2i1i1f1i1f2f2f1f1i1f3f3f1i")  # must match MaterialBlock's std140 layout exactly


def _pack_material_ubo(item_data, tinted=True):
    # Tint only applies when a color was chosen AND there's a mask to
    # apply it through; tinted=False forces it off (see _get_material_ubo).
    tint = item_data.get("tint_color") if tinted and item_data.get("tint_mask_texture") is not None else None
    tint_rgb = tuple(tint) if tint is not None else (1.0, 1.0, 1.0)
    emissive = tuple(item_data.get("emissive", (0.0, 0.0, 0.0)))
    pan1 = tuple(item_data.get("normal_pan1_speed", (0.0, 0.0)))
    pan2 = tuple(item_data.get("normal_pan2_speed", (0.0, 0.0)))
    return _MATERIAL_UBO_STRUCT.pack(
        emissive[0], emissive[1], emissive[2], 0.0,
        float(item_data.get("metallic", 0.0)),
        min(float(item_data.get("roughness", 1.0)), 1.0),
        float(item_data.get("specular_strength", 1.0)),
        float(item_data.get("base_alpha", 1.0)),
        int(item_data.get("has_texture", 0)),
        int(item_data.get("has_metallic_roughness_texture", 0)),
        _ALPHA_MODE_TO_INT.get(item_data.get("alpha_mode", "OPAQUE"), 0),
        float(item_data.get("alpha_cutoff", 0.5)),
        int(item_data.get("has_normal_texture", 0)),
        float(item_data.get("normal_scale", 1.0)),
        float(pan1[0]), float(pan1[1]),
        float(pan2[0]), float(pan2[1]),
        float(item_data.get("normal_uv2_scale", 1.0)),
        _REFLECTION_MODE_TO_INT.get(item_data.get("reflection_mode", "cheap"), 0),
        float(item_data.get("normal_uv_scale", 1.0)),
        0.0, 0.0, 0.0,  # _pad_material2a/b/c - unused, see MaterialBlock's own comment
        float(tint_rgb[0]), float(tint_rgb[1]), float(tint_rgb[2]),
        1 if tint is not None else 0,
    )


def _get_material_ubo(ctx, item_data, tinted=True):
    """tinted=False returns a separate cached buffer with the player-color
    tint forced off (a worn hat shares its owner's material but has its own
    UVs, which the tint mask doesn't line up with). Everything below is
    about the normal, tinted=True buffer.

    Returns item_data's own MaterialBlock uniform buffer, building
    and caching it (on item_data itself, keyed "_material_ubo") the
    first time this object is ever drawn - every field packed into it
    (see _pack_material_ubo) is fixed at load time (metallic/roughness/
    emissive/alpha_mode/... - see model_loader.py's _extract_material)
    and never changes again, so building+uploading it once and just
    RE-BINDING (no data upload) on every later draw is exactly the same
    fix already applied to skeletal bone matrices - see skeletal_
    shader.py's upload_bone_matrices/bind_bone_matrices. Confirmed via
    CPU profiling that rebuilding this data into a fresh Python dict and
    writing it as 9 separate uniforms, every object, every frame, was a
    real, avoidable cost once per-frame draw counts grew into the
    hundreds (a multi-material level, not a handful of props)."""
    key = "_material_ubo" if tinted else "_material_ubo_untinted"
    ubo = item_data.get(key)
    if ubo is None:
        ubo = ctx.buffer(_pack_material_ubo(item_data, tinted))
        item_data[key] = ubo
    return ubo


def invalidate_material_ubo(item_data):
    """Drops item_data's cached material buffers so the next draw repacks
    them - call after changing a field that's normally fixed at load time
    (e.g. tint_color, see Scene.set_skeletal_tint)."""
    for key in ("_material_ubo", "_material_ubo_untinted"):
        ubo = item_data.pop(key, None)
        if ubo is not None:
            ubo.release()


def bind_untinted_material(ctx, item_data):
    """Rebinds item_data's material block with the color tint off - for the
    hat drawn right after a tinted base mesh (see Scene._draw_hat)."""
    _get_material_ubo(ctx, item_data, tinted=False).bind_to_uniform_block(MATERIAL_UBO_BINDING)


def bind_material(prog, item_data, model_matrix, view_proj):
    """Binds everything that varies PER OBJECT - transform, material
    factors/textures, lightmap. view_proj is projection * view, from
    this frame's own bind_frame_uniforms(prog, ...) call (same camera,
    same prog) - see that function's own docstring for why this is
    passed in rather than a `camera` object each call would have to
    re-derive view/projection from again.

    Material factors (metallic/roughness/emissive/alpha_mode/...) come
    from item_data's own cached MaterialBlock buffer (see _get_material_
    ubo) - only u_mvp/u_model genuinely change per draw (the camera
    moves every frame; a dynamic object's own transform can too)."""
    mvp = view_proj * model_matrix
    _write_uniform(prog, "u_mvp", mvp.to_bytes())
    _write_uniform(prog, "u_model", model_matrix.to_bytes())

    _get_material_ubo(prog.ctx, item_data).bind_to_uniform_block(MATERIAL_UBO_BINDING)

    _bind_material_textures(prog, item_data)
    _bind_lightmap(prog, item_data)


def bind_transform_only(prog, model_matrix, view_proj):
    """bind_material's per-object part alone (u_mvp / u_model): for an object drawn right after one
    with an identical material (same textures, same material buffer), which is already bound."""
    _write_uniform(prog, "u_mvp", (view_proj * model_matrix).to_bytes())
    _write_uniform(prog, "u_model", model_matrix.to_bytes())


def bind_environment(prog, sky_color, ground_color):
    """Binds the hemisphere ambient uniforms - see Scene.
    environment_sky_color/environment_ground_color (set by
    add_equirect_skybox/add_skybox) and the fragment shader's
    hemisphere_ambient computation. Call this once per frame, same as
    bind_point_lights - this doesn't vary between objects."""
    _write_uniform(prog, "u_sky_color", tuple(sky_color))
    _write_uniform(prog, "u_ground_color", tuple(ground_color))


def bind_reflection_environment(prog, texture, exposure=1.0, is_hdr=False):
    """Binds the scene's own equirectangular skybox texture (Scene.
    equirect_skybox_texture) as a crude reflection source for near-
    mirror-smooth materials (roughness < 0.15 - see FRAGMENT_SHADER_
    BODY's env_reflection block/sample_reflection_env) - a separate
    call from bind_environment's own hemisphere-ambient uniforms, even
    though both come from the same add_equirect_skybox call, because
    they need genuinely different data (this needs the actual texture +
    its exposure/HDR-ness to SAMPLE a direction; bind_environment only
    ever needs its two precomputed AVERAGE colors).

    texture=None (no equirect skybox loaded this scene, or a caller that
    only ever set up Scene.add_skybox's older cubemap-face skybox
    instead - that one has no single 2D texture this can sample the same
    way) clears u_has_reflection_env to 0 rather than leaving whatever
    was bound last frame, so a scene that never calls add_equirect_
    skybox at all sees this feature fully, harmlessly disabled. Call
    once per frame per program, same as bind_environment/bind_point_
    lights - this doesn't vary between objects either."""
    if texture is None:
        _write_uniform(prog, "u_has_reflection_env", 0)
        return
    texture.use(location=TEX_UNIT_REFLECTION_ENV)
    _write_uniform(prog, "u_reflection_env", TEX_UNIT_REFLECTION_ENV)
    _write_uniform(prog, "u_reflection_exposure", float(exposure))
    _write_uniform(prog, "u_reflection_is_hdr", 1 if is_hdr else 0)
    _write_uniform(prog, "u_has_reflection_env", 1)


def bind_ssr_textures(prog, color_texture, depth_texture):
    """Binds Scene.enable_screen_space_reflections' own per-frame grab
    of the already-rendered opaque scene's color+depth, if this scene
    has one - what an "ssr" (u_reflection_mode=1) material's own
    ssr_reflect ray-marches instead of the plain skybox bind_
    reflection_environment provides (see MaterialBlock's own
    u_reflection_mode comment).

    color_texture=None (no Scene.enable_screen_space_reflections call at
    all this scene, OR this draw is the MAIN pass before this frame's
    grab exists yet - see Scene._render_scene's own comment) clears
    u_has_scene_grab to 0 rather than leaving whatever was bound last
    frame - a material set to "ssr" then just falls back to the cheap/
    skybox path instead (see ssr_reflect's own short-circuit), never
    reads stale/undefined data. depth_texture is only ever None exactly
    when color_texture is too - both come from the same grab. Call once
    per frame per program, same as bind_reflection_environment/bind_
    environment/bind_point_lights."""
    if color_texture is None:
        _write_uniform(prog, "u_has_scene_grab", 0)
        return
    color_texture.use(location=TEX_UNIT_SSR_COLOR)
    depth_texture.use(location=TEX_UNIT_SSR_DEPTH)
    _write_uniform(prog, "u_scene_color", TEX_UNIT_SSR_COLOR)
    _write_uniform(prog, "u_scene_depth", TEX_UNIT_SSR_DEPTH)
    _write_uniform(prog, "u_has_scene_grab", 1)


def bind_point_lights(prog, point_lights):
    """
    Binds point-light uniforms for this frame - position, color
    (pre-multiplied by intensity), and falloff radius only. No shadow
    data: point lights are always unshadowed in real time now (see
    module docstring).

    Safe to call either once per frame (all objects share the same
    lights) or per-object (see Scene._nearest_point_lights - real-time-
    lit objects each rebind the MAX_POINT_LIGHTS lights nearest to
    THEIR OWN position, since a scene can have more lights than the
    real-time cap but any one object can only ever be near a few of
    them at once).

    point_lights: list of dicts shaped like Scene.add_point_light()
    produces (each needs "position", "color", "intensity", "radius").
    """
    lights = point_lights[:MAX_POINT_LIGHTS]

    _write_uniform(prog, "u_num_point_lights", len(lights))

    positions = []
    colors = []
    radii = []

    for light in lights:
        positions.extend(light["position"])
        colors.extend(c * light["intensity"] for c in light["color"])
        radii.append(light["radius"])

    def _padded(values, length):
        return values + [0.0] * (length - len(values))

    if _has_uniform(prog, "u_point_light_pos"):
        prog["u_point_light_pos"].write(
            np.array(_padded(positions, MAX_POINT_LIGHTS * 3), dtype=np.float32).tobytes()
        )
    if _has_uniform(prog, "u_point_light_color"):
        prog["u_point_light_color"].write(
            np.array(_padded(colors, MAX_POINT_LIGHTS * 3), dtype=np.float32).tobytes()
        )
    if _has_uniform(prog, "u_point_light_radius"):
        prog["u_point_light_radius"].write(
            np.array(_padded(radii, MAX_POINT_LIGHTS), dtype=np.float32).tobytes()
        )


def bind_probe_irradiance(prog, irradiance):
    """u_probe_irradiance - see that uniform's own comment. Call once
    PER OBJECT (not once per frame - this genuinely varies with world
    position), right alongside that object's own bind_point_lights call
    (see Scene._render_scene/_render_transparent_objects).

    irradiance: a length-3 iterable (Scene._sample_probe_irradiance's
    return value - a numpy array, but any length-3 iterable works)."""
    _write_uniform(prog, "u_probe_irradiance", tuple(float(c) for c in irradiance))