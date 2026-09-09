"""
GPU skinning shaders for skeletal meshes.

Reuses pbr_shader.py's FRAGMENT_SHADER_HEADER/BODY verbatim - only the
vertex stage changes for skinned meshes (positions/normals get bone-
blended before everything else proceeds identically to the rigid path).
This is deliberate: duplicating the fragment shader would mean any
future lighting fix (bias tuning, a new light type, etc.) needs to be
applied twice and can silently drift apart - exactly the class of bug
that already happened once in this project (MAX_SHADOW_POINT_LIGHTS
disagreeing between two copies) and was fixed by never letting two
copies of the same logic exist in the first place.

Bone matrices are supplied as ONE WHOLE ARRAY (prog["u_bone_matrices"].write(...)),
never via per-index bracket names - see pbr_shader.py's docstring for
why that distinction matters in moderngl.

MAX_BONES lives only in the vertex stage (the reused fragment shader has
no bone data at all), so it doesn't compete with the fragment shader's
existing register budget (cascades, point lights, lightmap) that caused
a real "Constant register limit exceeded" crash earlier in this project.
64 is a comfortable default for a single humanoid-scale rig; each mat4
in the array still costs registers on legacy compilers, so raise this
only if you've verified it compiles on your target hardware.
"""

import glm

from Modules.Graphics.pbr_shader import FRAGMENT_SHADER_HEADER, FRAGMENT_SHADER_BODY

MAX_BONES = 64

SKELETAL_VERTEX_HEADER = f"""
#version 330
#define MAX_BONES {MAX_BONES}
"""

SKELETAL_VERTEX_BODY = """
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_bone_matrices[MAX_BONES];

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;
in vec2 in_lightmap_uv;
in ivec4 in_joints;
in vec4 in_weights;

out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out vec2 v_lightmap_uv;

void main() {
    // Standard linear blend skinning - up to 4 bones per vertex,
    // weighted. Normal uses mat3(skin_matrix) rather than a proper
    // inverse-transpose: correct for uniformly-scaled bones, which is
    // effectively every game rig in practice (non-uniform bone scaling
    // is rare and usually avoided deliberately since it also breaks
    // other things like physics colliders).
    mat4 skin_matrix =
        in_weights.x * u_bone_matrices[in_joints.x] +
        in_weights.y * u_bone_matrices[in_joints.y] +
        in_weights.z * u_bone_matrices[in_joints.z] +
        in_weights.w * u_bone_matrices[in_joints.w];

    vec4 skinned_position = skin_matrix * vec4(in_position, 1.0);
    vec3 skinned_normal = mat3(skin_matrix) * in_normal;

    v_position = (u_model * skinned_position).xyz;
    v_normal = mat3(transpose(inverse(u_model))) * skinned_normal;
    v_color = in_color;
    v_uv = in_uv;
    v_lightmap_uv = in_lightmap_uv;
    gl_Position = u_mvp * skinned_position;
}
"""

SKELETAL_VERTEX_SHADER = SKELETAL_VERTEX_HEADER + SKELETAL_VERTEX_BODY

# Reused verbatim from pbr_shader.py - see module docstring for why.
SKELETAL_FRAGMENT_SHADER = FRAGMENT_SHADER_HEADER + FRAGMENT_SHADER_BODY


SKELETAL_SHADOW_VERTEX_HEADER = f"""
#version 330
#define MAX_BONES {MAX_BONES}
"""

SKELETAL_SHADOW_VERTEX_BODY = """
uniform mat4 u_light_mvp;
uniform mat4 u_bone_matrices[MAX_BONES];

in vec3 in_position;
in ivec4 in_joints;
in vec4 in_weights;

void main() {
    mat4 skin_matrix =
        in_weights.x * u_bone_matrices[in_joints.x] +
        in_weights.y * u_bone_matrices[in_joints.y] +
        in_weights.z * u_bone_matrices[in_joints.z] +
        in_weights.w * u_bone_matrices[in_joints.w];

    // u_light_mvp is already the combined light_vp * model matrix (same
    // convention as the rigid shadow_program in scene.py) - skinning
    // happens in the mesh's own local space first, same as in_position
    // would be for a rigid mesh, then the single combined matrix takes
    // it the rest of the way to light clip space.
    gl_Position = u_light_mvp * skin_matrix * vec4(in_position, 1.0);
}
"""

SKELETAL_SHADOW_VERTEX_SHADER = SKELETAL_SHADOW_VERTEX_HEADER + SKELETAL_SHADOW_VERTEX_BODY

SKELETAL_SHADOW_FRAGMENT_SHADER = """
#version 330
void main() {}
"""


def create_skeletal_program(ctx):
    return ctx.program(vertex_shader=SKELETAL_VERTEX_SHADER, fragment_shader=SKELETAL_FRAGMENT_SHADER)


def create_skeletal_shadow_program(ctx):
    return ctx.program(vertex_shader=SKELETAL_SHADOW_VERTEX_SHADER, fragment_shader=SKELETAL_SHADOW_FRAGMENT_SHADER)


def bind_bone_matrices(prog, bone_matrices):
    """bone_matrices: list of glm.mat4 (e.g. from
    Skeleton.compute_bone_matrices()), any length up to MAX_BONES -
    padded with identity matrices past that. Whole-array write, not
    per-index bracket names - see module docstring."""
    padded = list(bone_matrices[:MAX_BONES])
    while len(padded) < MAX_BONES:
        padded.append(glm.mat4(1.0))

    if "u_bone_matrices" in prog:
        prog["u_bone_matrices"].write(b"".join(m.to_bytes() for m in padded))