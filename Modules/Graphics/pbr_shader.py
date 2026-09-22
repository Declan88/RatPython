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
TEX_UNIT_SHADOW_START = 2  # cascades occupy TEX_UNIT_SHADOW_START..+cascade_count-1
MAX_SHADOW_CASCADES = 3

MAX_POINT_LIGHTS = 4  # total point lights the real-time shader evaluates, all unshadowed

# Free since real-time point-light shadows were removed (they used to
# occupy this range).
TEX_UNIT_LIGHTMAP = TEX_UNIT_SHADOW_START + MAX_SHADOW_CASCADES  # = 5

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
TEX_UNIT_MOVABLE_SHADOW_START = TEX_UNIT_LIGHTMAP + 1  # = 6, occupies 6..8

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

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out vec2 v_lightmap_uv;

void main() {
    v_position = (u_model * vec4(in_position, 1.0)).xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_color = in_color;
    v_uv = in_uv;
    v_lightmap_uv = in_lightmap_uv;
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
};

uniform sampler2D u_texture;
uniform sampler2D u_metallic_roughness_texture;

uniform sampler2D u_shadow_maps[NUM_CASCADES];
uniform mat4 u_light_mvps[NUM_CASCADES];
uniform float u_cascade_splits[NUM_CASCADES];
uniform int u_has_shadows;

// Movable-only cascades (dynamic/skeletal casters only - see
// TEX_UNIT_MOVABLE_SHADOW_START's own comment for why this is a
// separate set from u_shadow_maps above, not a reuse of it).
uniform sampler2D u_movable_shadow_maps[NUM_CASCADES];
uniform mat4 u_movable_light_mvps[NUM_CASCADES];
uniform float u_movable_cascade_splits[NUM_CASCADES];
uniform int u_has_movable_shadows;

uniform int u_num_point_lights;
uniform vec3 u_point_light_pos[MAX_POINT_LIGHTS];
uniform vec3 u_point_light_color[MAX_POINT_LIGHTS];
uniform float u_point_light_radius[MAX_POINT_LIGHTS];

uniform sampler2D u_lightmap;
uniform int u_has_lightmap;

// Hemisphere ("skylight"-style) ambient - see bind_environment() and
// Scene.add_equirect_skybox/add_skybox for where these come from.
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in vec2 v_lightmap_uv;

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
float calculate_shadow(vec3 world_pos, float view_depth, vec3 normal, vec3 light_dir) {
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

int get_movable_cascade_index(float view_depth) {
    if (view_depth < u_movable_cascade_splits[0]) return 0;
    if (view_depth < u_movable_cascade_splits[1]) return 1;
    // See get_cascade_index's own comment - same fix, same reason.
    if (view_depth < u_movable_cascade_splits[2]) return 2;
    return 2;
}

// Same normal-offset-bias + 3x3 PCF as calculate_shadow above, just
// against the movable-only cascade set - see TEX_UNIT_MOVABLE_SHADOW_
// START's own comment for why a static/lightmapped surface uses this
// instead of calculate_shadow.
float calculate_movable_shadow(vec3 world_pos, float view_depth, vec3 normal, vec3 light_dir) {
    if (u_has_movable_shadows == 0 || view_depth <= 0.0) return 0.0;

    int cascade = get_movable_cascade_index(view_depth);

    float offset_scale[NUM_CASCADES] = float[NUM_CASCADES](0.02, 0.05, 0.1);
    vec3 offset_pos = world_pos + normal * offset_scale[cascade];

    vec4 light_space = u_movable_light_mvps[cascade] * vec4(offset_pos, 1.0);
    if (light_space.w <= 0.00001) return 0.0;

    vec3 shadow_coord = (light_space.xyz / light_space.w) * 0.5 + 0.5;
    if (any(lessThan(shadow_coord, vec3(0.0))) || any(greaterThan(shadow_coord, vec3(1.0)))) return 0.0;

    float shadow = 0.0;
    vec2 texel_size = vec2(1.0 / SHADOW_MAP_RESOLUTION);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(shadow_coord.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (shadow_coord.z > texture(u_movable_shadow_maps[cascade], uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

// Point lights are always unshadowed in real time - see the module
// docstring for why (shadowed contributions are baked instead).
// Blinn-Phong: diffuse is plain Lambertian (albedo * NdotL, no PI
// normalization - Source's model isn't energy-conserving, it's an
// artist-tuned look), specular is pow(NdotH, shininess) tinted by
// specular_color, gated by NdotL so it doesn't light backfacing spots.
vec3 calculate_point_light(int i, vec3 N, vec3 V, vec3 albedo, float shininess, vec3 specular_color, vec3 world_pos) {
    vec3 light_vec = u_point_light_pos[i] - world_pos;
    float dist = length(light_vec);
    vec3 L = light_vec / max(dist, 0.0001);
    vec3 H = normalize(V + L);

    float radius = max(u_point_light_radius[i], 0.01);
    float falloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
    float atten = (falloff * falloff) / (dist * dist + 1.0);

    float NdotL = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);

    vec3 diffuse = albedo * NdotL;
    vec3 specular = specular_color * spec * NdotL * u_specular_strength;

    return (diffuse + specular) * u_point_light_color[i] * atten;
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
    vec3 V = normalize(u_eye_pos - v_position);
    vec3 L = normalize(u_light_dir);
    vec3 H = normalize(V + L);

    vec4 tex_sample = u_has_texture == 1 ? texture(u_texture, v_uv) : vec4(v_color, 1.0);
    vec3 raw_albedo = tex_sample.rgb;
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
        float movable_shadow = calculate_movable_shadow(v_position, view_depth, N, L);
        point_light_sum = baked * (1.0 - movable_shadow);
    } else {
        int num_points = min(u_num_point_lights, MAX_POINT_LIGHTS);
        for (int i = 0; i < num_points; i++) {
            point_light_sum += calculate_point_light(i, N, V, albedo, shininess, specular_color, v_position);
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
    vec3 color = direct_light + point_light_sum + ambient + u_emissive_packed.xyz;

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
    if "MaterialBlock" in prog:
        prog["MaterialBlock"].binding = MATERIAL_UBO_BINDING
    return prog


def create_program(ctx):
    return bind_material_block(
        ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    )


def _write_uniform(prog, name, value):
    """Set a uniform whether it needs .write() (matrices) or .value = (scalars/vectors)."""
    if name not in prog:
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
    ):
        tex = item_data.get(key)
        uniform_name = f"u_{key}"
        if tex and uniform_name in prog:
            tex.use(location=unit)
            prog[uniform_name].value = unit


def _bind_shadow_uniforms(prog, shadow_manager):
    if shadow_manager is None:
        return

    cascades = shadow_manager.depth_textures[:MAX_SHADOW_CASCADES]
    for i, tex in enumerate(cascades):
        tex.use(location=TEX_UNIT_SHADOW_START + i)

    _write_uniform(prog, "u_has_shadows", 1)
    if "u_shadow_maps" in prog:
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


def _bind_movable_shadow_uniforms(prog, movable_shadow_manager):
    """Same shape as _bind_shadow_uniforms, for the SECOND (movable-
    casters-only) cascade set a static/lightmapped surface samples
    instead of the main one - see TEX_UNIT_MOVABLE_SHADOW_START and
    main()'s own u_has_lightmap branch for why this needs to be a
    wholly separate set of textures/uniforms rather than reusing
    u_shadow_maps."""
    if movable_shadow_manager is None:
        return

    cascades = movable_shadow_manager.depth_textures[:MAX_SHADOW_CASCADES]
    for i, tex in enumerate(cascades):
        tex.use(location=TEX_UNIT_MOVABLE_SHADOW_START + i)

    _write_uniform(prog, "u_has_movable_shadows", 1)
    if "u_movable_shadow_maps" in prog:
        units = tuple(TEX_UNIT_MOVABLE_SHADOW_START + i for i in range(MAX_SHADOW_CASCADES))
        prog["u_movable_shadow_maps"].value = units

    splits = list(movable_shadow_manager.splits)
    while len(splits) < MAX_SHADOW_CASCADES:
        splits.append(movable_shadow_manager.far)
    _write_uniform(
        prog,
        "u_movable_cascade_splits",
        np.array(splits[:MAX_SHADOW_CASCADES], dtype=np.float32).tobytes(),
    )

    _write_uniform(
        prog,
        "u_movable_light_mvps",
        b"".join(m.to_bytes() for m in movable_shadow_manager.light_mvps[:MAX_SHADOW_CASCADES]),
    )


def _bind_lightmap(prog, item_data):
    texture = item_data.get("lightmap_texture")
    if texture is not None and "u_lightmap" in prog:
        texture.use(location=TEX_UNIT_LIGHTMAP)
        prog["u_lightmap"].value = TEX_UNIT_LIGHTMAP
        _write_uniform(prog, "u_has_lightmap", 1)
    else:
        _write_uniform(prog, "u_has_lightmap", 0)


def bind_frame_uniforms(
    prog, camera, light_dir, shadow_manager=None,
    light_color=(1.0, 1.0, 1.0), light_intensity=2.0, movable_shadow_manager=None
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

    movable_shadow_manager: the SECOND, movable-casters-only cascade
    set (see TEX_UNIT_MOVABLE_SHADOW_START/_bind_movable_shadow_
    uniforms) - what a static/lightmapped surface samples to receive a
    real-time shadow from a moving object without double-counting its
    already-baked static-on-static shadowing. Optional/None for a
    caller with no such second shadow map (there's nothing for a
    static-only or dynamic-only draw pass to gain from it either way)."""
    view = camera.get_view_matrix()
    view_proj = camera.get_projection_matrix() * view

    _write_uniform(prog, "u_view_matrix", view.to_bytes())
    _write_uniform(prog, "u_light_dir", tuple(light_dir))
    _write_uniform(prog, "u_light_color", tuple(light_color))
    _write_uniform(prog, "u_light_intensity", float(light_intensity))
    _write_uniform(prog, "u_eye_pos", tuple(camera.position))
    # Explicit default before _bind_shadow_uniforms (which only ever
    # WRITES u_has_shadows=1, never clears it) - this call happens once
    # per frame now rather than once per object, so a shadow_manager
    # that goes from set to None between frames needs this reset here
    # or the shader would keep reading last frame's stale "1" forever.
    _write_uniform(prog, "u_has_shadows", 0)
    _bind_shadow_uniforms(prog, shadow_manager)
    _write_uniform(prog, "u_has_movable_shadows", 0)
    _bind_movable_shadow_uniforms(prog, movable_shadow_manager)

    return view_proj


_MATERIAL_UBO_STRUCT = struct.Struct("<4f4f2i1i1f")  # must match MaterialBlock's std140 layout exactly


def _pack_material_ubo(item_data):
    emissive = tuple(item_data.get("emissive", (0.0, 0.0, 0.0)))
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
    )


def _get_material_ubo(ctx, item_data):
    """Returns item_data's own MaterialBlock uniform buffer, building
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
    ubo = item_data.get("_material_ubo")
    if ubo is None:
        ubo = ctx.buffer(_pack_material_ubo(item_data))
        item_data["_material_ubo"] = ubo
    return ubo


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


def bind_environment(prog, sky_color, ground_color):
    """Binds the hemisphere ambient uniforms - see Scene.
    environment_sky_color/environment_ground_color (set by
    add_equirect_skybox/add_skybox) and the fragment shader's
    hemisphere_ambient computation. Call this once per frame, same as
    bind_point_lights - this doesn't vary between objects."""
    _write_uniform(prog, "u_sky_color", tuple(sky_color))
    _write_uniform(prog, "u_ground_color", tuple(ground_color))


def bind_point_lights(prog, point_lights):
    """
    Binds point-light uniforms for this frame - position, color
    (pre-multiplied by intensity), and falloff radius only. No shadow
    data: point lights are always unshadowed in real time now (see
    module docstring).

    Call this ONCE PER FRAME, before your object render loop - none of
    this data changes between draw calls.

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

    if "u_point_light_pos" in prog:
        prog["u_point_light_pos"].write(
            np.array(_padded(positions, MAX_POINT_LIGHTS * 3), dtype=np.float32).tobytes()
        )
    if "u_point_light_color" in prog:
        prog["u_point_light_color"].write(
            np.array(_padded(colors, MAX_POINT_LIGHTS * 3), dtype=np.float32).tobytes()
        )
    if "u_point_light_radius" in prog:
        prog["u_point_light_radius"].write(
            np.array(_padded(radii, MAX_POINT_LIGHTS), dtype=np.float32).tobytes()
        )