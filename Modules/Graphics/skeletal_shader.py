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
// [0] mvp, [1] model, [2] normal matrix (upper-left 3x3), one write per draw - see pbr_shader._write_object_uniforms
uniform mat4 u_object[3];
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

    v_position = (u_object[1] * skinned_position).xyz;
    v_normal = mat3(u_object[2]) * skinned_normal;
    v_color = in_color;
    v_uv = in_uv;
    v_lightmap_uv = in_lightmap_uv;
    // See this file's own v_tangent declaration comment above - a fixed
    // placeholder, never actually sampled against for a skeletal object.
    v_tangent = vec4(1.0, 0.0, 0.0, 1.0);
    gl_Position = u_object[0] * skinned_position;
}
"""

SKELETAL_VERTEX_SHADER = SKELETAL_VERTEX_HEADER + SKELETAL_VERTEX_BODY

# The Source-engine "dissolve" death effect (Modules/Weapons/damage_classes.py's own Zap) -
# skeletal-only (a static wall never dissolves), so this is spliced into a COPY of
# FRAGMENT_SHADER_BODY, not the shared constant itself (see this file's own module
# docstring on why that constant stays one source of truth for the LIGHTING it computes -
# this doesn't touch any of that, it only ever discards or tints the fragment AFTER
# lighting already ran, so there's nothing here that could drift from the static/dynamic
# pipeline's own copy of the same shared body). u_dissolve_amount defaults to 0.0 (moderngl
# zero-initializes every uniform), which skips this block entirely - so every OTHER
# skeletal object drawn with this same program (everyone not currently dissolving) is
# completely unaffected; only the object whose OWN per-draw uniform write (see
# bind_dissolve) sets it above 0 ever tints or discards at all.
#
# This is a real port of what Source actually does, read from the engine's own client code
# (source-sdk-2013's game/client/c_entitydissolve.cpp - ValveSoftware/source-sdk-2013 on
# GitHub), not a guess: C_EntityDissolve::ClientThink sets the whole model's render colour
# to (1 - fadeInPercentage) * effectColor, i.e. the model tints from its normal lit colour
# DOWN TO FLAT BLACK as the effect ramps in (kRenderTransColor fully replaces the model's
# shading with that flat colour - there is no per-pixel cutout pattern in Source's own
# implementation at all). Every crackling spark/glow/tesla-arc the player actually sees is
# a completely separate particle effect spawned around the character's hitboxes each frame
# from C_EntityDissolve::DrawModel/DoSparks (see Modules/Particles/dissolve_sparks.py and
# RemotePlayer's own periodic spawning while self._dissolving - nothing to do with this
# shader), and only once GetModelFadeOutPercentage() drops does the model's ALPHA fade out
# to make it disappear.
#
# Source achieves that alpha fade with a real blend-mode switch this engine's skeletal
# pipeline doesn't have (every skeletal object always draws fully opaque - see _render_
# scene's own comment on why). A dithered/stippled discard - a fixed per-screen-pixel
# threshold pattern, the fragment discarded once u_dissolve_amount's "vanish" phase passes
# that pixel's threshold - reaches the same "gradually disappears" result without needing a
# real alpha blend, at the cost of a visible dither pattern up close instead of a smooth
# blend (the standard trick for fading an opaque-pipeline object out; not something Source's
# own code does, since it doesn't need to).
_DISSOLVE_UNIFORMS = """
uniform float u_dissolve_amount;   // 0 = untouched, 1 = fully gone

float dissolve_hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}
"""
# Fractions of u_dissolve_amount's own 0-1 range (see RemotePlayer.DISSOLVE_SECONDS) spent
# turning black vs. then vanishing - tuned so the body reads as fully black for a beat before
# it starts disappearing, same shape as Source's own separate fade-in/fade-out windows.
_DISSOLVE_BLACKEN_END = 0.35
_DISSOLVE_VANISH_START = 0.55
_DISSOLVE_TARGET = "    fragColor = vec4(color, u_alpha_mode == 2 ? alpha : 1.0);\n}"
_DISSOLVE_SNIPPET = f"""
    if (u_dissolve_amount > 0.0) {{
        // Colour ramps to flat black - see this block's own comment above for exactly where
        // this comes from in Source's own client code.
        float blacken = clamp(u_dissolve_amount / {_DISSOLVE_BLACKEN_END}, 0.0, 1.0);
        color *= (1.0 - blacken);
        // Then dissolves away via a per-pixel dither threshold (see this block's own comment
        // on why a discard stipple stands in for Source's real alpha blend here).
        float vanish = clamp((u_dissolve_amount - {_DISSOLVE_VANISH_START}) / (1.0 - {_DISSOLVE_VANISH_START}), 0.0, 1.0);
        if (vanish > 0.0 && dissolve_hash(gl_FragCoord.xy) < vanish) discard;
    }}
    fragColor = vec4(color, u_alpha_mode == 2 ? alpha : 1.0);
}}"""
assert _DISSOLVE_TARGET in FRAGMENT_SHADER_BODY, "pbr_shader.FRAGMENT_SHADER_BODY's ending changed shape"
_SKELETAL_FRAGMENT_BODY = FRAGMENT_SHADER_BODY.replace(_DISSOLVE_TARGET, _DISSOLVE_SNIPPET, 1)

SKELETAL_FRAGMENT_SHADER = FRAGMENT_SHADER_HEADER + _DISSOLVE_UNIFORMS + _SKELETAL_FRAGMENT_BODY


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
    if hasattr(bone_matrices, "gpu_bytes"):        # a pose_batch.BoneMatrices: already packed
        return bone_matrices.gpu_bytes()
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


def bind_dissolve(prog, obj):
    """Writes u_dissolve_amount for obj's draw - 0 (untouched) for the overwhelming majority
    of skeletal objects, every frame, which never have obj["dissolve_amount"] set at all (see
    RemotePlayer's own dissolve state machine, the only thing that ever sets it) - this still
    has to be written per-draw regardless, since the shader program is SHARED across every
    skeletal object: without explicitly zeroing it for everyone else, whichever object last
    left it non-zero would leak its black tint/vanish onto the next object drawn with this
    program."""
    if "u_dissolve_amount" not in prog:
        return
    prog["u_dissolve_amount"].value = float(obj.get("dissolve_amount") or 0.0)
