"""
Simplified Cook-Torrance PBR shader + cascaded-shadow-aware material binding.

Changes from the original version:
- GLSL: shadow bias scaling per cascade replaced with an array lookup
  instead of an if/else chain. No functional change.
- GLSL: texture/sampler unit numbers and the shadow-map resolution used
  for PCF texel size are now named constants at the top of the fragment
  shader, so the "2048" and "3" scattered through the code only need to
  match CascadedShadowMap.resolution / cascade_count in one place.
- Python: bind_material() split into small helpers (_write_uniform,
  _bind_material_textures, _bind_shadow_uniforms) instead of one function
  doing uniform building, texture binding, and shadow wiring all inline.
  This also removes the redundant double-set of "u_has_shadows" (it was
  set to 0 in the base dict, then unconditionally overwritten to 1 inside
  the shadow branch).
- Python: texture unit numbers are named constants instead of bare 0/1/2.
- Behavior and the public function signatures (create_program,
  bind_material) are unchanged, so this is a drop-in replacement.

Known fragility carried over from the original (not changed here, flagging
for awareness): the shader hardcodes 3 shadow-cascade samplers and a PCF
texel size derived from a 2048 shadow-map resolution. If you ever construct
CascadedShadowMap with cascade_count != 3 or a different resolution, the
shader's assumptions (u_shadow_maps[3], u_cascade_splits[3], texel_size)
will silently mismatch it. Worth keeping these two files' constants in sync,
or passing resolution in as a uniform, if that ever changes.
"""

import moderngl
import numpy as np

# Fixed texture units used when binding materials.
TEX_UNIT_ALBEDO = 0
TEX_UNIT_METALLIC_ROUGHNESS = 1
TEX_UNIT_SHADOW_START = 2  # cascades occupy TEX_UNIT_SHADOW_START..+cascade_count-1
MAX_SHADOW_CASCADES = 3

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

FRAGMENT_SHADER = """
#version 330

#define NUM_CASCADES 3
#define SHADOW_MAP_RESOLUTION 2048.0

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
    float bias = max(0.0005, 0.003 * (1.0 - clamp(dot(normal, light_dir), 0.0, 1.0)));
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
    vec3 ambient = vec3(0.025) * albedo;
    vec3 color = direct_light + ambient + u_emissive;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
"""


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
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
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
