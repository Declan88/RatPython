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
without hitting shader register/texture-unit limits). Only the
directional light (CascadedShadowMap) still casts real-time dynamic
shadows, which matters for moving objects a baked lightmap can't cover.

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

import moderngl
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

uniform float u_metallic;
uniform float u_roughness;
uniform float u_specular_strength;
uniform vec3 u_emissive;

uniform sampler2D u_texture;
uniform int u_has_texture;

uniform sampler2D u_metallic_roughness_texture;
uniform int u_has_metallic_roughness_texture;

uniform sampler2D u_shadow_maps[NUM_CASCADES];
uniform mat4 u_light_mvps[NUM_CASCADES];
uniform float u_cascade_splits[NUM_CASCADES];
uniform int u_has_shadows;

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
    vec3 N = normalize(v_normal);
    vec3 V = normalize(u_eye_pos - v_position);
    vec3 L = normalize(u_light_dir);
    vec3 H = normalize(V + L);

    vec3 raw_albedo = u_has_texture == 1 ? texture(u_texture, v_uv).rgb : v_color;
    vec3 albedo = pow(max(raw_albedo, vec3(0.0)), vec3(2.2));

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
    float shadow_attenuation = 1.0 - calculate_shadow(v_position, view_depth, N, L);

    vec3 diffuse = albedo * NdotL;
    vec3 specular = specular_color * spec * NdotL * u_specular_strength;
    vec3 direct_light = (diffuse + specular) * u_light_color * u_light_intensity * shadow_attenuation;

    vec3 point_light_sum = vec3(0.0);
    if (u_has_lightmap == 1) {
        // Lightmaps store pure incoming light (irradiance), same as
        // Source's model - they get MULTIPLIED by the surface's own
        // albedo, not added raw.
        point_light_sum = texture(u_lightmap, v_lightmap_uv).rgb * albedo;
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
    vec3 color = direct_light + point_light_sum + ambient + u_emissive;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
"""

FRAGMENT_SHADER = FRAGMENT_SHADER_HEADER + FRAGMENT_SHADER_BODY


def create_program(ctx):
    return ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)


def _write_uniform(prog, name, value):
    """Set a uniform whether it needs .write() (matrices) or .value = (scalars/vectors)."""
    if name not in prog:
        return
    if isinstance(value, (bytes, bytearray)):
        prog[name].write(value)
    else:
        prog[name].value = value


def _bind_material_textures(prog, item_data):
    for key, unit in (
        ("texture", TEX_UNIT_ALBEDO),
        ("metallic_roughness_texture", TEX_UNIT_METALLIC_ROUGHNESS),
    ):
        tex = item_data.get(key)
        uniform_name = f"u_{key}"
        if tex and uniform_name in prog:
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
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


def _bind_lightmap(prog, item_data):
    texture = item_data.get("lightmap_texture")
    if texture is not None and "u_lightmap" in prog:
        texture.use(location=TEX_UNIT_LIGHTMAP)
        prog["u_lightmap"].value = TEX_UNIT_LIGHTMAP
        _write_uniform(prog, "u_has_lightmap", 1)
    else:
        _write_uniform(prog, "u_has_lightmap", 0)


def bind_material(
    prog, item_data, model_matrix, camera, light_dir, shadow_manager=None,
    light_color=(1.0, 1.0, 1.0), light_intensity=2.0
):
    """light_color/light_intensity: the directional (sun) light's own
    color and brightness multiplier - see Scene.light_color/
    Scene.light_intensity. Defaults match this shader's previous
    hardcoded behavior (implicitly white at a fixed 2.0 multiplier)
    exactly, for any caller that doesn't pass them."""
    view = camera.get_view_matrix()
    mvp = camera.get_projection_matrix() * view * model_matrix

    base_uniforms = {
        "u_mvp": mvp.to_bytes(),
        "u_model": model_matrix.to_bytes(),
        "u_view_matrix": view.to_bytes(),
        "u_light_dir": tuple(light_dir),
        "u_light_color": tuple(light_color),
        "u_light_intensity": float(light_intensity),
        "u_eye_pos": tuple(camera.position),
        "u_metallic": item_data.get("metallic", 0.0),
        "u_roughness": min(item_data.get("roughness", 1.0), 1.0),
        "u_specular_strength": float(item_data.get("specular_strength", 1.0)),
        "u_emissive": tuple(item_data.get("emissive", (0.0, 0.0, 0.0))),
        "u_has_texture": item_data.get("has_texture", 0),
        "u_has_metallic_roughness_texture": item_data.get(
            "has_metallic_roughness_texture", 0
        ),
        "u_has_shadows": 0,
    }
    for name, value in base_uniforms.items():
        _write_uniform(prog, name, value)

    _bind_material_textures(prog, item_data)
    _bind_shadow_uniforms(prog, shadow_manager)
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