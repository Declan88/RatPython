"""
Simplified Cook-Torrance PBR shader + cascaded-shadow-aware material binding.

Point light support:
- GLSL: adds up to MAX_POINT_LIGHTS point lights with inverse-square-ish
  windowed attenuation, of which the first MAX_SHADOW_POINT_LIGHTS can
  cast real shadows via PointShadowMap (see point_shadow_module.py).
  Point shadows use 6 separate 2D depth textures per light (not a true
  cubemap - see point_shadow_module.py docstring for why), and the
  fragment shader picks the right face per-fragment by the major axis
  of the direction from the light to the surface.
- Python: bind_point_lights(prog, point_lights) binds all point-light
  uniforms and shadow textures ONCE PER FRAME (not per object, since none
  of that data changes between draw calls in a frame) - call it before
  your object render loop, alongside/after bind_material() per object.

IMPORTANT - array uniform binding: every array uniform in this file is
assigned as ONE WHOLE ARRAY (prog["name"].value = tuple(...), or
prog["name"].write(bytes)), never via per-index bracket names like
prog["name[3]"]. moderngl typically only registers a single active
uniform per array (usually "name[0]"), so "name[3]" in prog silently
evaluates False and a per-index write just never happens. An earlier
version of bind_point_lights did exactly this and was a real, silent
bug: shadow textures got bound to the correct GL texture units fine,
but the shader's samplers were never told which units to read from, so
no point-light shadow ever actually appeared. Whole-array assignment
(the same approach _bind_shadow_uniforms already used for the
directional cascades) is the pattern proven to work here - don't
reintroduce per-index bracket names for arrays in this file.

IMPORTANT - keeping Python and GLSL light counts in sync: NUM_CASCADES,
MAX_POINT_LIGHTS, and MAX_SHADOW_POINT_LIGHTS are injected directly into
the GLSL source via FRAGMENT_SHADER_HEADER below, rather than being
hardcoded a second time as #define values in the shader text. This is
deliberate - these two numbers drifting apart is exactly what happened
when MAX_SHADOW_POINT_LIGHTS was bumped to 20 in Python while the GLSL
#define was still hand-written as 1: Python then tried to bind uniform
data for 20 shadow-casting lights (20 * 6 = 120 texture units, on top
of the 5 already used for albedo/metallic-roughness/cascades) against
a shader that had only actually declared array space for 6. If you need
to change MAX_SHADOW_POINT_LIGHTS, changing the constant below is now
the only edit required - just remember the texture-unit budget note
next to it.

Same "keep in sync" caveat still applies to SHADOW_MAP_RESOLUTION and
POINT_SHADOW_RESOLUTION below, which are NOT derived from a shared
Python constant (they just have to match CascadedShadowMap's and
PointShadowMap's resolution arguments by convention). Passing resolution
in as a uniform instead would remove this last manual-sync point, if
that's ever worth doing.
"""

import moderngl
import numpy as np
import glm

# Fixed texture units used when binding materials.
TEX_UNIT_ALBEDO = 0
TEX_UNIT_METALLIC_ROUGHNESS = 1
TEX_UNIT_SHADOW_START = 2  # cascades occupy TEX_UNIT_SHADOW_START..+cascade_count-1
MAX_SHADOW_CASCADES = 3

# Point lights occupy the texture units right after the cascades.
TEX_UNIT_POINT_SHADOW_START = TEX_UNIT_SHADOW_START + MAX_SHADOW_CASCADES  # = 5
MAX_POINT_LIGHTS = 30       # total point lights the shader will evaluate

# How many of those get real shadows. Each one costs 6 texture units
# (one per cube face). GL 3.3 only guarantees 16 total in the fragment
# stage, and TEX_UNIT_POINT_SHADOW_START already uses 5 of them, so:
#   5 + MAX_SHADOW_POINT_LIGHTS * 6
# must stay comfortably under your actual hardware's limit - check via
# ctx.info['GL_MAX_TEXTURE_IMAGE_UNITS'] before raising this. And even
# where the texture-unit budget allows it, every one of these lights
# does 6 real shadow-frustum render passes per frame (or per bake, for
# static geometry) - treat this as a scarce resource, not a light
# count. 2-4 is a reasonable ceiling for most scenes.
MAX_SHADOW_POINT_LIGHTS = 27

# Injected directly into the GLSL #defines below so Python and GLSL
# can never drift apart the way NUM_CASCADES historically could.
FRAGMENT_SHADER_HEADER = f"""
#version 330

#define NUM_CASCADES {MAX_SHADOW_CASCADES}
#define SHADOW_MAP_RESOLUTION 2048.0

#define MAX_POINT_LIGHTS {MAX_POINT_LIGHTS}
#define MAX_SHADOW_POINT_LIGHTS {MAX_SHADOW_POINT_LIGHTS}
#define POINT_SHADOW_RESOLUTION 1024.0
"""

VERTEX_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;

void main() {
    v_position = (u_model * vec4(in_position, 1.0)).xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_color = in_color;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER_BODY = """
uniform vec3 u_light_dir;
uniform vec3 u_eye_pos;
uniform mat4 u_view_matrix;

uniform float u_metallic;
uniform float u_roughness;
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
uniform int u_point_light_has_shadow[MAX_POINT_LIGHTS];

uniform sampler2D u_point_shadow_faces[MAX_SHADOW_POINT_LIGHTS * 6];
uniform mat4 u_point_light_mvps[MAX_SHADOW_POINT_LIGHTS * 6];

in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;

out vec4 fragColor;
const float PI = 3.14159265359;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a2 = max(pow(roughness, 4.0), 0.00001);
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 0.0001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float k = pow(roughness + 1.0, 2.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

int get_cascade_index(float view_depth) {
    if (view_depth < u_cascade_splits[0]) return 0;
    if (view_depth < u_cascade_splits[1]) return 1;
    return 2;
}

float calculate_shadow(vec3 world_pos, float view_depth, vec3 normal, vec3 light_dir) {
    if (u_has_shadows == 0 || view_depth <= 0.0) return 0.0;

    int cascade = get_cascade_index(view_depth);
    vec4 light_space = u_light_mvps[cascade] * vec4(world_pos, 1.0);
    if (light_space.w <= 0.00001) return 0.0;

    vec3 shadow_coord = (light_space.xyz / light_space.w) * 0.5 + 0.5;
    if (any(lessThan(shadow_coord, vec3(0.0))) || any(greaterThan(shadow_coord, vec3(1.0)))) return 0.0;

    float bias_scale[NUM_CASCADES] = float[NUM_CASCADES](1.0, 1.5, 2.0);
    float bias = -0.0002;
    bias *= bias_scale[cascade];

    float shadow = 0.0;
    vec2 texel_size = vec2(1.0 / SHADOW_MAP_RESOLUTION);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(shadow_coord.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (shadow_coord.z - bias > texture(u_shadow_maps[cascade], uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

// Picks which of the 6 baked faces a point-light ray falls into.
// Order must match PointShadowMap.FACE_DIRECTIONS in point_shadow_module.py:
// 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
int get_cube_face(vec3 dir) {
    vec3 a = abs(dir);
    if (a.x >= a.y && a.x >= a.z) return dir.x > 0.0 ? 0 : 1;
    if (a.y >= a.x && a.y >= a.z) return dir.y > 0.0 ? 2 : 3;
    return dir.z > 0.0 ? 4 : 5;
}

float calculate_point_shadow(int light_index, vec3 world_pos) {
    vec3 dir = world_pos - u_point_light_pos[light_index];
    int slot = light_index * 6 + get_cube_face(dir);

    vec4 light_space = u_point_light_mvps[slot] * vec4(world_pos, 1.0);
    if (light_space.w <= 0.00001) return 0.0;

    vec3 shadow_coord = (light_space.xyz / light_space.w) * 0.5 + 0.5;
    if (any(lessThan(shadow_coord, vec3(0.0))) || any(greaterThan(shadow_coord, vec3(1.0)))) return 0.0;

    float bias = -.00005;
    float shadow = 0.0;
    vec2 texel_size = vec2(1.0 / POINT_SHADOW_RESOLUTION);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec2 uv = clamp(shadow_coord.xy + vec2(x, y) * texel_size, vec2(0.001), vec2(0.999));
            if (shadow_coord.z - bias > texture(u_point_shadow_faces[slot], uv).r) {
                shadow += 1.0;
            }
        }
    }
    return shadow / 9.0;
}

vec3 calculate_point_light(int i, vec3 N, vec3 V, vec3 albedo, float rough, float metal, vec3 F0, vec3 world_pos) {
    vec3 light_vec = u_point_light_pos[i] - world_pos;
    float dist = length(light_vec);
    vec3 L = light_vec / max(dist, 0.0001);
    vec3 H = normalize(V + L);

    float radius = max(u_point_light_radius[i], 0.01);
    float falloff = clamp(1.0 - pow(dist / radius, 4.0), 0.0, 1.0);
    float atten = (falloff * falloff) / (dist * dist + 1.0);

    float NDF = distributionGGX(N, H, rough);
    float G = geometrySmith(N, V, L, rough);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metal);
    float NdotL = max(dot(N, L), 0.0);

    float shadow_atten = 1.0;
    if (u_point_light_has_shadow[i] == 1) {
        shadow_atten = 1.0 - calculate_point_shadow(i, world_pos);
    }

    return (kD * albedo / PI + specular) * u_point_light_color[i] * NdotL * atten * shadow_atten;
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

    vec3 F0 = mix(vec3(0.04), albedo, metal);
    float NDF = distributionGGX(N, H, rough);
    float G = geometrySmith(N, V, L, rough);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metal);
    float NdotL = max(dot(N, L), 0.0);

    float view_depth = -(u_view_matrix * vec4(v_position, 1.0)).z;
    float shadow_attenuation = 1.0 - calculate_shadow(v_position, view_depth, N, L);

    vec3 direct_light = (kD * albedo / PI + specular) * vec3(2.0) * NdotL * shadow_attenuation;

    vec3 point_light_sum = vec3(0.0);
    int num_points = min(u_num_point_lights, MAX_POINT_LIGHTS);
    for (int i = 0; i < num_points; i++) {
        point_light_sum += calculate_point_light(i, N, V, albedo, rough, metal, F0, v_position);
    }

    vec3 ambient = vec3(0.025) * albedo;
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
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
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


def bind_material(
    prog, item_data, model_matrix, camera, light_dir, shadow_manager=None
):
    view = camera.get_view_matrix()
    mvp = camera.get_projection_matrix() * view * model_matrix

    base_uniforms = {
        "u_mvp": mvp.to_bytes(),
        "u_model": model_matrix.to_bytes(),
        "u_view_matrix": view.to_bytes(),
        "u_light_dir": tuple(light_dir),
        "u_eye_pos": tuple(camera.position),
        "u_metallic": item_data.get("metallic", 0.0),
        "u_roughness": min(item_data.get("roughness", 1.0), 1.0),
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


def bind_point_lights(prog, point_lights):
    """
    Bind all point-light uniforms and shadow textures for this frame.

    Call this ONCE PER FRAME, before your object render loop - unlike
    bind_material(), none of this data changes between draw calls, so
    re-binding it per object would just waste uniform/texture-unit
    upload bandwidth for no benefit.

    Every array here is assigned as one whole array, not via per-index
    bracket names - see the module docstring for why that distinction
    matters (it was the actual cause of point shadows never appearing).

    point_lights: list of dicts shaped like Scene.add_point_light()
    produces, each with "position", "color", "intensity", "radius",
    and optionally "shadow_map" (a PointShadowMap instance or None).
    """
    lights = point_lights[:MAX_POINT_LIGHTS]

    _write_uniform(prog, "u_num_point_lights", len(lights))

    positions = []
    colors = []
    radii = []
    has_shadow = [0] * MAX_POINT_LIGHTS

    # Full-length flat arrays for every possible shadow slot, even ones
    # unused this frame - array uniforms need a value for every element,
    # not just the currently-active ones.
    shadow_units = [TEX_UNIT_POINT_SHADOW_START] * (MAX_SHADOW_POINT_LIGHTS * 6)
    shadow_mvps = [glm.mat4(1.0) for _ in range(MAX_SHADOW_POINT_LIGHTS * 6)]

    shadow_slot = 0

    for i, light in enumerate(lights):
        positions.extend(light["position"])
        colors.extend(c * light["intensity"] for c in light["color"])
        radii.append(light["radius"])

        shadow_map = light.get("shadow_map")
        if shadow_map is not None and shadow_slot < MAX_SHADOW_POINT_LIGHTS:
            has_shadow[i] = 1
            for face in range(6):
                unit = TEX_UNIT_POINT_SHADOW_START + shadow_slot * 6 + face
                shadow_map.live_textures[face].use(location=unit)
                shadow_units[shadow_slot * 6 + face] = unit
                shadow_mvps[shadow_slot * 6 + face] = shadow_map.light_mvps[face]
            shadow_slot += 1

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

    _write_uniform(prog, "u_point_light_has_shadow", tuple(has_shadow))

    if "u_point_shadow_faces" in prog:
        prog["u_point_shadow_faces"].value = tuple(shadow_units)

    if "u_point_light_mvps" in prog:
        prog["u_point_light_mvps"].write(b"".join(m.to_bytes() for m in shadow_mvps))