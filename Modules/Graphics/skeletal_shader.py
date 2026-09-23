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

Bone matrices live in a std140 UNIFORM BUFFER (block "BoneBlock"), one
buffer per skeletal object, uploaded ONCE per frame (see
upload_bone_matrices) and merely re-bound for every draw that frame
(bind_bone_matrices) - the main shadow cascades, the movable-only
cascades and the color pass all skin from the same pose. This used to
be a plain program-uniform array (prog["u_bone_matrices"].write(...))
rewritten on every draw, which on Intel Arc integrated graphics made
the FIRST such write each frame block inside the driver for up to
~250ms (measured: p99 37ms) - visible as random hard stutters - on top
of re-packing 64 mat4s in Python once per draw call.

MAX_BONES lives only in the vertex stage (the reused fragment shader has
no bone data at all), so it doesn't compete with the fragment shader's
existing register budget (cascades, point lights, lightmap) that caused
a real "Constant register limit exceeded" crash earlier in this project.
64 is a comfortable default for a single humanoid-scale rig; each mat4
in the array still costs registers on legacy compilers, so raise this
only if you've verified it compiles on your target hardware.
"""

import glm

from Modules.Graphics.pbr_shader import FRAGMENT_SHADER_HEADER, FRAGMENT_SHADER_BODY, bind_material_block

MAX_BONES = 64

# Uniform-buffer binding point shared by every skeletal program (only one
# object's bones are bound at any instant, so one point is enough).
BONE_UBO_BINDING = 0

SKELETAL_VERTEX_HEADER = f"""
#version 330
#define MAX_BONES {MAX_BONES}
"""

SKELETAL_VERTEX_BODY = """
uniform mat4 u_mvp;
uniform mat4 u_model;
layout(std140) uniform BoneBlock { mat4 u_bone_matrices[MAX_BONES]; };

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
// Fixed placeholder, not real per-vertex tangent data - see pbr_
// shader.py's FRAGMENT_SHADER_BODY (shared verbatim by this program -
// see this file's own module docstring) for why v_tangent still needs
// to exist here even though no skeletal mesh currently has one: a
// skinned rig has no in_tangent attribute or _compute_tangents call of
// its own (normal mapping was only ever added for static level
// geometry - see model_loader.py's own comment), so any GLSL "in" the
// shared fragment shader declares still needs SOME matching vertex
// "out" for this program to link at all, regardless of whether that
// data is ever meaningful. Harmless: no skeletal object currently sets
// has_normal_texture=1, so the fragment shader's tangent-consuming
// branch (apply_normal_map) never actually executes for one.
out vec4 v_tangent;

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
    // See this file's own v_tangent declaration comment above - a fixed
    // placeholder, never actually sampled against for a skeletal object.
    v_tangent = vec4(1.0, 0.0, 0.0, 1.0);
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
layout(std140) uniform BoneBlock { mat4 u_bone_matrices[MAX_BONES]; };

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


def _bind_bone_block(prog):
    if "BoneBlock" in prog:
        prog["BoneBlock"].binding = BONE_UBO_BINDING
    return prog


def create_skeletal_program(ctx):
    # Also binds MaterialBlock (pbr_shader.py's bind_material_block) -
    # this program reuses FRAGMENT_SHADER_BODY verbatim (see module
    # docstring), so it declares that same block too.
    return bind_material_block(_bind_bone_block(
        ctx.program(vertex_shader=SKELETAL_VERTEX_SHADER, fragment_shader=SKELETAL_FRAGMENT_SHADER)
    ))


def create_skeletal_shadow_program(ctx):
    return _bind_bone_block(
        ctx.program(vertex_shader=SKELETAL_SHADOW_VERTEX_SHADER, fragment_shader=SKELETAL_SHADOW_FRAGMENT_SHADER)
    )


def _pack_bone_matrices(bone_matrices):
    padded = list(bone_matrices[:MAX_BONES])
    while len(padded) < MAX_BONES:
        padded.append(glm.mat4(1.0))
    return b"".join(m.to_bytes() for m in padded)


def upload_bone_matrices(ctx, obj):
    """Writes obj["bone_matrices"] (list of glm.mat4, any length up to
    MAX_BONES, identity-padded past that) into obj's own uniform buffer,
    creating it on first use. Call once per frame per object after the
    pose is computed (Scene.update does) - NOT once per draw. Skipped
    entirely if the pose list is the very same object as last upload
    (Scene.update only rebinds obj["bone_matrices"] when it recomputes)."""
    bones = obj["bone_matrices"]
    if obj.get("_bones_uploaded") is bones and obj.get("bone_ubo") is not None:
        return
    data = _pack_bone_matrices(bones)
    ubo = obj.get("bone_ubo")
    if ubo is None:
        obj["bone_ubo"] = ctx.buffer(data, dynamic=True)
    else:
        ubo.orphan()
        ubo.write(data)
    obj["_bones_uploaded"] = bones


def bind_bone_matrices(obj):
    """Makes obj's bone uniform buffer the active BoneBlock source for
    subsequent draws - a cheap bind, no data upload (see
    upload_bone_matrices)."""
    obj["bone_ubo"].bind_to_uniform_block(BONE_UBO_BINDING)
