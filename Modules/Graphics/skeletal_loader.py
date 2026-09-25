"""
glTF skeletal mesh loading.

Reads a glb's skin (joint hierarchy + inverse bind matrices), animation
clips, per-vertex skinning attributes (JOINTS_0/WEIGHTS_0), and material
(base color/metallic/roughness/emissive + textures) directly via raw
glTF JSON parsing, independent of trimesh (which doesn't expose skinning
data at all - same reason gltf_lights.py and model_loader.py's lightmap
UV reader go straight to the raw glTF instead of through trimesh).

Material extraction needs a live moderngl context to upload textures
(load_skinned_glb's ctx parameter) - pass ctx=None to get geometry/
animation/material-factor data without any texture upload, e.g. for a
pure data-inspection tool with no GL context available.

SCOPE, stated explicitly rather than silently assumed:
- One skin per file (first one used if a file somehow has more, which
  is unusual). Any number of mesh primitives IS supported - they're
  merged into one combined mesh (a skin applies to every primitive of
  its node's mesh in glTF, not per-primitive, so this is just normal
  multi-material-slot export, not an edge case) - but only ONE
  material (the first primitive's) is used for the whole merged result,
  same simplification model_loader.py's own multi-object flatten
  already makes for a non-skinned mesh. True per-primitive materials
  would need per-primitive draw calls or a texture atlas, neither
  implemented here.
- LINEAR and STEP keyframe interpolation only. CUBICSPLINE (which stores
  in-tangent/value/out-tangent triples per keyframe instead of a single
  value) is a real, valid glTF interpolation mode but roughly doubles
  the sampling code - clips using it will print a warning and fall back
  to treating the values as LINEAR, which will look wrong rather than
  crash, so check the console if an imported animation looks off.
- A joint's parent is found by walking the whole node graph once and
  matching against other joints in the SAME skin. If a joint's real
  parent lies outside the skin's joint list (e.g. an armature root
  above the actual bone hierarchy), that joint is treated as a root for
  pose computation - this is a simplification: most standard rig
  exports (Blender's glTF exporter included) include the full bone
  hierarchy within skin.joints, so this rarely matters in practice, but
  it's worth knowing about if a character's root bone doesn't seem to
  inherit expected parent motion.
- Material textures are assumed to reference TEXCOORD_0 (the same UV
  set already read for the mesh) - glTF technically allows a texture to
  reference a different UV channel via its own "texCoord" index, not
  handled here since skinned meshes in practice only ever have one UV
  set (no lightmap UV - see add_skeletal's docstring for why).
"""

import io
import json
import struct
import bisect
from pathlib import Path

import numpy as np
import moderngl
import glm
from PIL import Image

# A glTF node named with one of these prefixes/suffixes (matched against
# the NODE's own name - what an artist names the object in Blender's
# outliner, not the mesh data-block name, same distinction model_loader.
# py's own COLLISION_ONLY_PREFIX convention makes for static meshes) is
# excluded from the merged mesh entirely - never loaded, not just drawn
# invisible. Built for exactly this repo's rat.glb: "physics_"/"_physics"
# for a physics-only collision proxy mesh baked into the same file as the
# visual one (present here as a node literally named "funnyrat_physics" -
# a SUFFIX, not a prefix, hence matching both), and "hat_" for a set of
# optional cosmetic hat variants (rat.glb has 8 of them) meant to default
# off rather than all render simultaneously stacked on the same head.
_HIDDEN_NODE_PREFIXES = ("hat_", "physics_")
_HIDDEN_NODE_SUFFIXES = ("_physics",)
# Hidden "hat_*" nodes aren't discarded: their geometry is kept as an optional
# add-on (see load_hat) that a Scene can draw on top of the base mesh. Each has
# its OWN material/texture, so it can't be merged into the base mesh.
_HAT_PREFIX = "hat_"


def list_hat_names(path):
    """Short names (prefix stripped, e.g. "cowboy") of the hat nodes in a
    skinned glb, in file order."""
    result = _read_glb_json_and_blob(path)
    if result is None:
        return []
    gltf, _ = result
    return [
        n["name"][len(_HAT_PREFIX):] for n in gltf.get("nodes", [])
        if n.get("mesh") is not None and (n.get("name") or "").lower().startswith(_HAT_PREFIX)
    ]


def _is_hidden_by_default(node_name):
    name = (node_name or "").lower()
    return name.startswith(_HIDDEN_NODE_PREFIXES) or name.endswith(_HIDDEN_NODE_SUFFIXES)


_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COMPONENT_COUNTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _read_accessor(gltf, blob, accessor_index):
    """Generic accessor reader - returns a numpy array shaped
    (count, component_count), or (count,) for SCALAR. Does not handle
    sparse accessors (prints a warning and returns None if encountered -
    rare for skinning data in practice)."""
    acc = gltf["accessors"][accessor_index]
    if acc.get("sparse"):
        print(f"[skeletal_loader] Sparse accessor {accessor_index} not supported, skipping.")
        return None

    dtype = _COMPONENT_DTYPES.get(acc["componentType"])
    count_per_elem = _TYPE_COMPONENT_COUNTS.get(acc["type"])
    if dtype is None or count_per_elem is None:
        print(f"[skeletal_loader] Unsupported accessor type/componentType at {accessor_index}, skipping.")
        return None

    view = gltf["bufferViews"][acc["bufferView"]]
    offset = (view.get("byteOffset", 0)) + (acc.get("byteOffset", 0))
    count = acc["count"] * count_per_elem

    data = np.frombuffer(blob, dtype=dtype, count=count, offset=offset).copy()

    # Normalized integer attributes (e.g. some WEIGHTS_0 exports use
    # normalized u8/u16 instead of float) need scaling to 0..1.
    if acc.get("normalized") and dtype != np.float32:
        max_val = np.iinfo(dtype).max
        data = data.astype(np.float32) / max_val

    if count_per_elem == 1:
        return data
    return data.reshape(-1, count_per_elem)


def _node_local_matrix(node):
    if "matrix" in node:
        return glm.mat4(*node["matrix"])

    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # gltf order: x, y, z, w
    scale = node.get("scale", [1.0, 1.0, 1.0])

    t = glm.translate(glm.mat4(1.0), glm.vec3(*translation))
    q = glm.quat(rotation[3], rotation[0], rotation[1], rotation[2])  # glm order: w, x, y, z
    r = glm.mat4_cast(q)
    s = glm.scale(glm.mat4(1.0), glm.vec3(*scale))
    return t * r * s


def _build_parent_map(gltf):
    """node_index -> parent_node_index, or -1 for a node with no parent
    (a root, or never referenced as a child)."""
    parents = {}
    for node_index, node in enumerate(gltf.get("nodes", [])):
        for child_index in node.get("children", []):
            parents[child_index] = node_index
    return parents


def _external_ancestor_matrix(node_idx, parent_map, gltf):
    """The combined world matrix of everything ABOVE node_idx in the
    glTF node graph - i.e. the transform that has to premultiply
    node_idx's own local matrix to place it correctly, whatever that
    ancestor chain happens to be (an armature object's own scale/
    translation/rotation sitting above the skeleton root is the case
    this exists for - see Joint.external_root_matrix - but this walks
    arbitrarily far up, not just one level). Returns identity if
    node_idx has no parent at all. Composed root-first (the topmost
    ancestor's local matrix leftmost) since world = parentWorld *
    childLocal, same convention _world_matrices' own walk uses."""
    chain = []
    current = parent_map.get(node_idx, -1)
    while current != -1:
        chain.append(current)
        current = parent_map.get(current, -1)

    matrix = glm.mat4(1.0)
    for ancestor_idx in reversed(chain):
        matrix = matrix * _node_local_matrix(gltf["nodes"][ancestor_idx])
    return matrix


class Joint:
    __slots__ = (
        "name", "node_index", "parent_joint_index", "local_bind_matrix",
        "inverse_bind_matrix", "external_root_matrix",
    )

    def __init__(self, name, node_index, parent_joint_index, local_bind_matrix,
                 inverse_bind_matrix, external_root_matrix=None):
        self.name = name
        self.node_index = node_index
        self.parent_joint_index = parent_joint_index  # index into Skeleton.joints, or -1
        self.local_bind_matrix = local_bind_matrix
        self.inverse_bind_matrix = inverse_bind_matrix
        # Only ever non-identity for a joint whose true glTF parent (see
        # _build_parent_map) lies OUTSIDE the skin's own joint list -
        # e.g. an armature OBJECT node sitting above the skeleton root
        # with its own scale/translation/rotation, a common result of a
        # Blender export where the armature object itself was scaled/
        # moved without "Apply Transform" first. That ancestor transform
        # is exactly what the file's exported inverse bind matrices were
        # computed against, so it has to be applied somewhere - see
        # _external_ancestor_matrix (where this is actually computed)
        # and Skeleton._world_matrices' root-joint case (where it gets
        # multiplied in) for why silently dropping it is wrong rather
        # than merely imprecise.
        self.external_root_matrix = external_root_matrix if external_root_matrix is not None else glm.mat4(1.0)


class AnimationChannel:
    __slots__ = ("joint_index", "path", "times", "values", "interpolation")

    def __init__(self, joint_index, path, times, values, interpolation):
        self.joint_index = joint_index
        self.path = path  # "translation" | "rotation" | "scale"
        # Converted from numpy once here at LOAD time (a one-off cost),
        # not kept as numpy arrays - sample() below runs on EVERY
        # candidate clip, for EVERY channel, EVERY frame (confirmed via
        # cProfile: this was the single hottest function in the whole
        # per-frame update, ~1.7ms of a ~2.1ms Scene.update on a single
        # skeletal object with an upper/lower blend split - see git
        # history around this comment for the profile). A plain Python
        # list + bisect beats np.searchsorted here because numpy's
        # per-call dispatch overhead dominates at these array sizes (a
        # clip's channel times list is typically tens of entries, not
        # thousands) - and per-component tuple arithmetic beats numpy
        # elementwise ops for the same reason on a 3- or 4-vector.
        self.times = [float(t) for t in times]  # seconds, ascending
        self.values = [tuple(float(x) for x in row) for row in values]  # (x,y,z) or (x,y,z,w) tuples
        self.interpolation = interpolation  # "LINEAR" | "STEP" (CUBICSPLINE falls back to LINEAR with a warning)

    def sample(self, time):
        times = self.times
        count = len(times)
        if count == 0:
            return None
        if time <= times[0]:
            idx = 0
            t = 0.0
        elif time >= times[-1]:
            idx = count - 2 if count > 1 else 0
            t = 1.0 if count > 1 else 0.0
        else:
            idx = bisect.bisect_right(times, time) - 1
            idx = max(0, min(idx, count - 2))
            span = times[idx + 1] - times[idx]
            t = 0.0 if span <= 0 else (time - times[idx]) / span

        if self.interpolation == "STEP":
            t = 0.0

        v0 = self.values[idx]
        v1 = self.values[min(idx + 1, len(self.values) - 1)]

        if self.path == "rotation":
            q0 = glm.quat(v0[3], v0[0], v0[1], v0[2])
            q1 = glm.quat(v1[3], v1[0], v1[1], v1[2])
            q = glm.slerp(q0, q1, t)
            return ("rotation", q)
        else:
            return (self.path, glm.vec3(
                v0[0] + (v1[0] - v0[0]) * t,
                v0[1] + (v1[1] - v0[1]) * t,
                v0[2] + (v1[2] - v0[2]) * t,
            ))


class AnimationClip:
    def __init__(self, name, channels):
        self.name = name
        self.channels = channels  # list[AnimationChannel]
        # Full authored keyframe range - no auto-trimming of a trailing
        # static hold (an earlier version of this detected and cut that
        # off automatically, which mattered when the source files had a
        # long static tail baked in; they've since been re-exported
        # trimmed at the source, so the raw range is already exactly the
        # intended loop and this stays a plain, predictable "however
        # many frames are actually in the file" duration).
        self.duration = max((c.times[-1] for c in channels if len(c.times) > 0), default=0.0)

    def sample_pose(self, joint_count, time):
        """Returns a list of length joint_count: local_translation (vec3
        or None), local_rotation (quat or None), local_scale (vec3 or
        None) per joint - None where this clip has no channel for that
        joint/path, meaning the joint's rest-pose value should be used
        instead (see Skeleton.compute_bone_matrices)."""
        translations = [None] * joint_count
        rotations = [None] * joint_count
        scales = [None] * joint_count

        for channel in self.channels:
            result = channel.sample(time)
            if result is None:
                continue
            path, value = result
            if path == "translation":
                translations[channel.joint_index] = value
            elif path == "rotation":
                rotations[channel.joint_index] = value
            elif path == "scale":
                scales[channel.joint_index] = value

        return translations, rotations, scales


def _extract_rotation(matrix):
    """Returns matrix's rotation component as a glm.quat, robust to
    uniform (or non-uniform) scale baked into it - unlike
    glm.quat_cast(glm.mat3(matrix)), which assumes its input already
    has unit-length basis columns and silently returns a WRONG
    (unnormalized garbage) quaternion otherwise. This matters here
    because a joint's accumulated WORLD matrix commonly does carry a
    scale factor with no rotation of its own mixed in - e.g. a skin's
    root joint scaled by a fixed unit-conversion factor via its own
    Joint.external_root_matrix (confirmed a real case, not just a
    theoretical one: rat.glb's own Pelvis root is scaled by 0.024) -
    which propagates into every descendant's world matrix multiplicatively
    even though each joint's own LOCAL matrix might be scale=1.
    glm.decompose handles this correctly by actually separating out
    scale before returning rotation, at the cost of also computing
    (and discarding) translation/skew/perspective this caller doesn't
    need - a worthwhile trade for correctness over the cheaper but
    unsafe quat_cast shortcut."""
    scale, rotation, translation = glm.vec3(), glm.quat(), glm.vec3()
    skew, perspective = glm.vec3(), glm.vec4()
    glm.decompose(matrix, scale, rotation, translation, skew, perspective)
    return rotation


class Skeleton:
    def __init__(self, joints, animations):
        self.joints = joints  # list[Joint], in skin.joints order
        self.animations = animations  # dict[name -> AnimationClip]
        # Pose snapshotting (see Scene._begin_pose_snapshot): last_pose is the
        # final local (translation, rotation, scale) per joint the previous
        # _world_matrices call produced - i.e. exactly what was on screen -
        # and _pose_snap, when set to (snapshot, weight), makes the next
        # walk blend from that snapshot toward the freshly computed pose.
        self.last_pose = None
        self._pose_snap = None
        self._bind_trs = {}

    def _empty_pose(self):
        joint_count = len(self.joints)
        return [None] * joint_count, [None] * joint_count, [None] * joint_count

    def _sample_clip(self, animation_name, time):
        """(translations, rotations, scales) - one list of length
        len(self.joints) each, per-joint None where neither this clip
        nor (below) the caller has anything for that joint - see
        AnimationClip.sample_pose. animation_name=None (or unknown)
        returns all-None, meaning "use bind pose" once _world_matrices
        below falls through to joint.local_bind_matrix."""
        clip = self.animations.get(animation_name)
        if clip is None:
            return self._empty_pose()
        return clip.sample_pose(len(self.joints), time)

    @staticmethod
    def _local_matrix(joint, t, r, s):
        """One joint's local transform from a (possibly partial - any of
        t/r/s may be None, meaning "use the bind pose's value for just
        this component") sampled pose, or the joint's bind pose outright
        if all three are None. Shared by _world_matrices' hierarchy walk
        and _sample_track's crossfade blending (which needs concrete
        local matrices, from both the outgoing and incoming clip, to
        decompose and interpolate - see its own docstring)."""
        if t is not None or r is not None or s is not None:
            tt = t if t is not None else glm.vec3(0.0)
            rr = r if r is not None else glm.quat(1.0, 0.0, 0.0, 0.0)
            ss = s if s is not None else glm.vec3(1.0)
            return glm.translate(glm.mat4(1.0), tt) * glm.mat4_cast(rr) * glm.scale(glm.mat4(1.0), ss)
        return joint.local_bind_matrix

    def _blend_pose_sets(self, from_pose, to_pose, weight):
        """Elementwise per-joint TRS blend between two already-sampled
        poses (translations, rotations, scales) triples - shared by
        _sample_track's two-clip crossfade and _sample_weighted's N-clip
        blend space below, since both ultimately need "combine two poses
        by a weight" as their base operation. Builds each joint's
        concrete local matrix via _local_matrix (bind-pose fallback
        already resolved, so there's always a real matrix on both sides
        even if one pose is None/missing a channel for this joint),
        decomposes each via glm.decompose, and interpolates the
        decomposed TRS components (translation/scale lerp, rotation
        slerp - matrices themselves can't be linearly interpolated,
        hence decomposing first) weighted by weight (0 = entirely
        from_pose, 1 = entirely to_pose)."""
        from_t, from_r, from_s = from_pose
        to_t, to_r, to_s = to_pose

        joint_count = len(self.joints)
        translations, rotations, scales = [None] * joint_count, [None] * joint_count, [None] * joint_count

        for i, joint in enumerate(self.joints):
            from_matrix = self._local_matrix(joint, from_t[i], from_r[i], from_s[i])
            to_matrix = self._local_matrix(joint, to_t[i], to_r[i], to_s[i])

            from_scale, from_rot, from_trans = glm.vec3(), glm.quat(), glm.vec3()
            glm.decompose(from_matrix, from_scale, from_rot, from_trans, glm.vec3(), glm.vec4())
            to_scale, to_rot, to_trans = glm.vec3(), glm.quat(), glm.vec3()
            glm.decompose(to_matrix, to_scale, to_rot, to_trans, glm.vec3(), glm.vec4())

            translations[i] = glm.mix(from_trans, to_trans, weight)
            rotations[i] = glm.slerp(from_rot, to_rot, weight)
            scales[i] = glm.mix(from_scale, to_scale, weight)

        return translations, rotations, scales

    def _sample_track(self, animation_name, time, prev_animation_name=None,
                       prev_time=0.0, blend_weight=1.0):
        """The per-track (lower or upper) pose sampler behind both
        compute_bone_matrices and compute_blended_bone_matrices: with no
        crossfade in progress (prev_animation_name=None, or blend_weight
        already at/past 1.0), this is just _sample_clip(animation_name,
        time) - the plain, non-blending case every existing caller still
        gets by default.

        With a crossfade in progress, samples BOTH clips (prev_
        animation_name at prev_time - frozen at whatever pose was
        showing the moment the transition started, NOT still advancing -
        and animation_name at time) and blends them via _blend_pose_sets
        (0 = entirely the outgoing pose, 1 = entirely the incoming one).
        This is what turns a hard instant cut between clips (the
        previous behavior - visible as a pop at every idle/walk state
        change) into a smooth transition."""
        cur_pose = self._sample_clip(animation_name, time)
        if prev_animation_name is None or blend_weight >= 1.0:
            return cur_pose

        prev_pose = self._sample_clip(prev_animation_name, prev_time)
        return self._blend_pose_sets(prev_pose, cur_pose, blend_weight)

    def _sample_weighted(self, weighted):
        """The blend-space counterpart to _sample_clip: weighted is a
        list of (clip_name, time, weight) - possibly several clips
        sampled at once (e.g. idle/walk/run's adjacent speed tiers, or a
        directional tier's two bracketing facing clips), each already at
        its own independently-advancing time (see Scene.update()'s
        locomotion_time - clips here are NOT phase-matched/restarted the
        way single-clip state switches used to be, since nothing here
        ever hard-cuts to begin with). Entries with a non-positive weight
        or an unknown clip name are dropped, and the remainder's weights
        renormalized to sum to 1 - a caller doesn't need to pre-normalize
        (or pre-filter for clips this skeleton might not actually have).

        Combines the survivors via _blend_pose_sets, folded in one at a
        time: after the first k entries, the running pose already
        represents their correctly-weighted combination (by induction),
        so blending it against entry k+1 with weight
        entry_weight / (cumulative_weight so far) yields the correct
        weighted combination of all k+1 - the standard incremental
        generalization of a two-way weighted blend to N clips (rotation
        is technically only an approximation of a true weighted average
        this way, since slerp isn't linear/order-independent, but it's
        the same practical trick real-time engines use, and is exact for
        translation/scale). Returns bind pose (via _empty_pose) if
        nothing survives filtering - e.g. every weight was ~0 or every
        clip name was unrecognized."""
        entries = [(name, time, w) for name, time, w in weighted if w > 0.0 and name in self.animations]
        if not entries:
            return self._empty_pose()
        if len(entries) == 1:
            name, time, _w = entries[0]
            return self._sample_clip(name, time)

        total = sum(w for _, _, w in entries)
        pose = self._sample_clip(entries[0][0], entries[0][1])
        running_weight = entries[0][2] / total
        for name, time, w in entries[1:]:
            running_weight += w / total
            blend = (w / total) / running_weight if running_weight > 0.0 else 0.0
            pose = self._blend_pose_sets(pose, self._sample_clip(name, time), blend)
        return pose

    def _world_matrices(self, translations, rotations, scales, rotation_offsets=None):
        """Shared hierarchy walk: given a per-joint local pose (any
        entry may be None, meaning "use this joint's bind-pose local
        transform instead" - see AnimationClip.sample_pose), returns the
        final list of glm.mat4 skinning matrices (world * inverse_bind),
        one per joint, ready to upload as the GPU skinning palette. Used
        by both compute_bone_matrices (single clip) and
        compute_blended_bone_matrices (two clips, picked per joint by a
        mask) - the walk itself doesn't care where each joint's local
        pose came from.

        rotation_offsets: optional {joint_index: glm.quat} map of
        COMPONENT-space correction rotations (see compute_blended_bone_
        matrices' own docstring for why/where this is used - upper-body
        pose corrections) - "component space" in the same sense Unreal's
        AnimGraph uses the term: fixed relative to this skeleton's OWN
        root (this method's implicit reference frame throughout - see
        world_matrix's own root case below), not the joint's immediate
        parent NOR the actual game-world/level axes. Deliberately not
        the joint's own local/parent-relative space: a joint's local
        axes are whatever its parent bone's rest orientation happens to
        leave them as, which for a bent/twisted rig bears no predictable
        relationship to the skeleton's own root orientation - tuning
        "-45 on Y" expecting a torso turn only reliably means that if Y
        is interpreted relative to the skeleton root regardless of the
        joint's own local orientation or its parent's current animated
        rotation. Also deliberately NOT true world/level space: this
        skeleton's root itself sits inside an outer model transform
        (obj["position"]/obj["rotation"] - see Scene._get_model_matrix)
        that can rotate independently of the skeleton every frame (the
        local player's own facing tracks the camera continuously) - a
        correction expressed in true level-space would then visibly
        fight that outer rotation instead of turning WITH the character
        as one rigid unit, which is what an authoring correction like
        this actually wants (confirmed the hard way: an earlier version
        of this feature used true absolute/level space and the
        correction visibly "un-rotated" itself relative to the body
        every time the player turned). Converted to the joint's actual
        local space here, during the walk, since that conversion needs
        the joint's PARENT component-space rotation (component_space =
        parent_component_space * local, so a component-space rotation R
        applied at this joint's own pivot is inverse(parent_rotation) *
        R * parent_rotation once expressed in local terms) - not
        available before this walk computes it."""
        world_cache = {}
        snap = self._pose_snap
        final_pose = [None] * len(self.joints)

        def world_matrix(i):
            if i in world_cache:
                return world_cache[i]

            joint = self.joints[i]
            parent_world = (
                joint.external_root_matrix if joint.parent_joint_index == -1
                # external_root_matrix is identity unless this joint's
                # real glTF parent lies outside the skin (e.g. an
                # armature object node with its own scale/translation/
                # rotation) - see Joint.external_root_matrix.
                else world_matrix(joint.parent_joint_index)
            )

            rotation = rotations[i]
            if rotation_offsets and i in rotation_offsets:
                base_rotation = (
                    rotation if rotation is not None
                    else _extract_rotation(joint.local_bind_matrix)
                )
                # NOT glm.quat_cast(glm.mat3(parent_world)) - quat_cast
                # requires unit-length basis columns and silently
                # returns garbage otherwise; parent_world's columns
                # carry whatever uniform scale this skeleton's root
                # external_root_matrix bakes in (e.g. a unit-conversion
                # factor - confirmed a real, not hypothetical, case:
                # rat.glb's own Pelvis root is scaled by 0.024). See
                # _extract_rotation for the decompose-based extraction
                # that handles this correctly.
                parent_rotation = _extract_rotation(parent_world)
                local_offset = glm.inverse(parent_rotation) * rotation_offsets[i] * parent_rotation
                rotation = local_offset * base_rotation

            t_i, s_i = translations[i], scales[i]
            if snap is not None:
                snap_pose, snap_weight = snap
                st, sr, ss = snap_pose[i]
                ct, cr, cs = self._concrete_trs(i, t_i, rotation, s_i)
                t_i = glm.mix(st, ct, snap_weight)
                rotation = glm.slerp(sr, cr, snap_weight)
                s_i = glm.mix(ss, cs, snap_weight)
                final_pose[i] = (t_i, rotation, s_i)
            else:
                final_pose[i] = self._concrete_trs(i, t_i, rotation, s_i)

            local = self._local_matrix(joint, t_i, rotation, s_i)
            world = parent_world * local

            world_cache[i] = world
            return world

        result = [world_matrix(i) * self.joints[i].inverse_bind_matrix for i in range(len(self.joints))]
        self.last_pose = final_pose
        return result

    def _concrete_trs(self, joint_index, t, r, s):
        """A joint's local (translation, rotation, scale) with every component
        resolved to a real value: any None component is identity, matching
        _local_matrix - except a joint with NO animated component at all,
        which is its bind pose (decomposed once and cached)."""
        if t is None and r is None and s is None:
            bind = self._bind_trs.get(joint_index)
            if bind is None:
                scale, rot, trans = glm.vec3(), glm.quat(), glm.vec3()
                glm.decompose(self.joints[joint_index].local_bind_matrix, scale, rot, trans, glm.vec3(), glm.vec4())
                bind = self._bind_trs[joint_index] = (trans, rot, scale)
            return bind
        return (t if t is not None else glm.vec3(0.0),
                r if r is not None else glm.quat(1.0, 0.0, 0.0, 0.0),
                s if s is not None else glm.vec3(1.0))

    def compute_bone_matrices(self, animation_name, time,
                               prev_animation_name=None, prev_time=0.0, blend_weight=1.0):
        """Returns a list of glm.mat4, one per joint (same order as
        self.joints), ready to upload as the GPU skinning palette.

        prev_animation_name/prev_time/blend_weight: optional crossfade -
        see _sample_track. Every existing caller that doesn't pass these
        gets the exact previous single-clip behavior (blend_weight's
        default of 1.0 always resolves to "just animation_name/time")."""
        translations, rotations, scales = self._sample_track(
            animation_name, time, prev_animation_name, prev_time, blend_weight
        )
        return self._world_matrices(translations, rotations, scales)

    def compute_bone_matrices_multi(self, weighted, prev_animation_name=None, prev_time=0.0, blend_weight=1.0):
        """The blend-space counterpart to compute_bone_matrices: weighted
        is a list of (clip_name, time, weight) - see _sample_weighted for
        exactly how those combine into one pose, instead of a single
        named clip. prev_animation_name/prev_time/blend_weight still
        crossfade from a single PREVIOUS clip exactly like
        compute_bone_matrices's own (blend_weight=1.0, the default,
        skips it entirely) - the same crossfade mechanism bridges a
        discrete state (e.g. a held jump/crouch pose) into or out of a
        continuous blend space without needing any special-casing at the
        transition itself."""
        pose = self._sample_weighted(weighted)
        if prev_animation_name is not None and blend_weight < 1.0:
            pose = self._blend_pose_sets(self._sample_clip(prev_animation_name, prev_time), pose, blend_weight)
        return self._world_matrices(*pose)

    def resolve_joint_indices(self, joint_names):
        """Returns the list of joint indices matching joint_names (by
        name), silently skipping any name not present in self.joints -
        shared by compute_joint_mask (root_joint_names) and callers that
        need the raw root INDICES themselves rather than the full
        descendant mask (e.g. add_skeletal's upper_rotation_offset,
        which must rotate only the roots themselves, not every joint the
        mask covers - rotating a descendant too would double-apply the
        correction on top of what it already inherits from its
        parent)."""
        name_to_index = {joint.name: i for i, joint in enumerate(self.joints)}
        return [name_to_index[name] for name in joint_names if name in name_to_index]

    def compute_joint_mask(self, root_joint_names):
        """Returns a list[bool] of length len(self.joints): True for
        every joint whose name is in root_joint_names, OR that is a
        transitive descendant (via parent_joint_index) of one that is.

        A list rather than a single root is deliberate, not just for
        flexibility: some rigs don't have one continuous "upper body"
        branch. rat.glb's Source/Valve Biped rig, for instance, parents
        its shoulder/arm/neck/head chain ("Spine4" and everything under
        it) directly to the PELVIS, as a sibling of the spine twist
        bones ("Spine"/"Spine1"/"Spine2") rather than a descendant of
        the last one - so "upper body" there is the union of two
        separate subtrees (["...Spine", "...Spine4"]), not one pivot
        joint's descendants. Unknown names in root_joint_names are
        silently ignored (contribute nothing to the mask) rather than
        raising, so a caller can pass a name list that's a superset of
        what a specific model's rig actually has."""
        root_indices = set(self.resolve_joint_indices(root_joint_names))

        mask = [False] * len(self.joints)
        for i, joint in enumerate(self.joints):
            # Walk up this joint's own parent chain - True if it passes
            # through any root index (including being one itself).
            j = i
            while j != -1:
                if j in root_indices:
                    mask[i] = True
                    break
                j = self.joints[j].parent_joint_index
        return mask

    def compute_blended_bone_matrices(self, lower_animation, lower_time,
                                       upper_animation, upper_time, upper_joint_mask,
                                       lower_prev_animation=None, lower_prev_time=0.0, lower_blend_weight=1.0,
                                       upper_prev_animation=None, upper_prev_time=0.0, upper_blend_weight=1.0,
                                       upper_rotation_offsets=None,
                                       upper_joint_mask_prev=None, mask_blend_weight=1.0):
        """Like compute_bone_matrices, but each joint's LOCAL pose comes
        from upper_animation (sampled at upper_time) where
        upper_joint_mask[joint_index] is True, and from lower_animation
        (sampled at lower_time) everywhere else - e.g. lower_animation
        driving legs/hips (idle/walk/run/jump/crouch) while
        upper_animation independently drives spine/arms/neck/head (a gun-
        holding pose), composited into one skeleton per frame. The
        hierarchy walk (world = parent_world * local) is unchanged and
        uniform - a joint's WORLD transform still depends on its
        parent's world transform regardless of which clip sourced its
        own local one, which is exactly how this kind of partial-body
        blend is supposed to work (an upper-body arm bone still follows
        the lower-body pelvis/spine root motion underneath it).

        upper_animation=None (no upper clip configured/playing yet)
        makes every joint fall back to lower_animation, matching
        compute_bone_matrices(lower_animation, lower_time) exactly -
        callers don't need a separate "no blending" code path.

        lower_prev_*/upper_prev_*: optional crossfade for EACH track
        independently - see _sample_track. Every existing caller that
        doesn't pass these gets the exact previous behavior (both
        blend_weight defaults are 1.0, meaning "just the current clip",
        same as compute_bone_matrices' own defaults).

        upper_rotation_offsets: an optional {joint_index: glm.quat}
        map of constant COMPONENT-space correction rotations, one
        independent value per joint - meant to be keyed by the upper-
        body mask's own ROOT joints (e.g. each clavicle, or Spine4 for a
        widened override mask - see add_skeletal's
        upper_rotation_offset_degrees, which is what actually builds
        this map), not every joint the mask covers: rotating a ROOT
        reorients its entire subtree for free via the normal parent-
        child world-matrix chain, so also rotating its descendants would
        double-apply the correction on top of what they already inherit.
        A practical knob for nudging a limb's authored rest orientation
        without touching the animation data itself (e.g. a pose authored
        on a rig with a different bind orientation than this skeleton's
        own) - keyed per-joint specifically so the LEFT and RIGHT sides
        of a symmetric rig can be corrected by different amounts
        independently, since a mirrored pose bug doesn't necessarily
        affect both sides equally (or at all). Component-space (fixed
        relative to this skeleton's own root, not the joint's immediate
        parent NOR the actual game world) specifically so a tuned value
        like "-45 on Y" reliably means the same turn relative to the
        character's own body regardless of whatever orientation the
        joint's parent bone happens to leave its local axes in, AND so
        the correction rotates rigidly WITH the character as the player
        turns instead of fighting that turn the way a true world/level-
        space correction would - see _world_matrices' own rotation_
        offsets docstring for the actual local-space conversion math,
        which needs the hierarchy walk itself (this method just forwards
        the map, unresolved, into that walk). None or an empty map (the
        default) skips this addition entirely, matching every caller
        before this param existed.

        upper_joint_mask_prev/mask_blend_weight: smooths a JOINT MASK
        change itself (see Scene.set_skeletal_upper_joint_mask) - e.g.
        widening the upper-body split from clavicles-only to Spine4 (and
        back) when a manual override starts/ends. Without this, any
        joint whose mask assignment actually differs between
        upper_joint_mask_prev and upper_joint_mask would hard-cut
        between an upper-sourced and lower-sourced pose the instant the
        mask changes - a real, confirmed pop distinct from (and on top
        of) upper_rotation_offsets' own crossfade, since the offset
        blend alone only smooths the CORRECTION, not the pose it's
        layered onto. For a joint whose mask assignment is UNCHANGED
        between the two masks, this has no effect at all (same selection
        as if upper_joint_mask_prev/mask_blend_weight were never passed)
        - only the joints actually transitioning between upper/lower
        control get slerped between their old and new pose source, using
        mask_blend_weight as the interpolation factor. upper_joint_mask_
        prev=None or mask_blend_weight>=1.0 (the defaults) skip this
        entirely, matching every caller before this param existed."""
        lower_pose = self._sample_track(
            lower_animation, lower_time, lower_prev_animation, lower_prev_time, lower_blend_weight
        )
        upper_pose = (
            None if upper_animation is None
            else self._sample_track(upper_animation, upper_time, upper_prev_animation, upper_prev_time, upper_blend_weight)
        )
        return self._composite_bone_matrices(
            lower_pose, upper_pose, upper_joint_mask, upper_rotation_offsets, upper_joint_mask_prev, mask_blend_weight
        )

    def compute_blended_bone_matrices_multi(self, lower_weighted, upper_weighted, upper_joint_mask,
                                             lower_prev_animation=None, lower_prev_time=0.0, lower_blend_weight=1.0,
                                             upper_prev_animation=None, upper_prev_time=0.0, upper_blend_weight=1.0,
                                             upper_rotation_offsets=None,
                                             upper_joint_mask_prev=None, mask_blend_weight=1.0):
        """The upper/lower-split counterpart to compute_bone_matrices_
        multi: each track's pose comes from a weighted list of clips (see
        _sample_weighted) instead of one named clip, then composited by
        upper_joint_mask exactly like compute_blended_bone_matrices - see
        its own docstring for the compositing/rotation-offset/mask-
        transition behavior, shared here via _composite_bone_matrices.
        upper_weighted=None falls back to the lower track's own pose for
        every joint, matching upper_animation=None's existing fallback in
        compute_blended_bone_matrices. lower_prev_animation/upper_prev_
        animation (each a single clip name, not a weighted list) still
        crossfade in/out of each track's blend space exactly like
        compute_bone_matrices_multi's own prev_animation_name."""
        lower_pose = self._sample_weighted(lower_weighted)
        if lower_prev_animation is not None and lower_blend_weight < 1.0:
            lower_pose = self._blend_pose_sets(
                self._sample_clip(lower_prev_animation, lower_prev_time), lower_pose, lower_blend_weight
            )

        if upper_weighted is None:
            upper_pose = None
        else:
            upper_pose = self._sample_weighted(upper_weighted)
            if upper_prev_animation is not None and upper_blend_weight < 1.0:
                upper_pose = self._blend_pose_sets(
                    self._sample_clip(upper_prev_animation, upper_prev_time), upper_pose, upper_blend_weight
                )

        return self._composite_bone_matrices(
            lower_pose, upper_pose, upper_joint_mask, upper_rotation_offsets, upper_joint_mask_prev, mask_blend_weight
        )

    def _composite_bone_matrices(self, lower_pose, upper_pose, upper_joint_mask,
                                  upper_rotation_offsets=None,
                                  upper_joint_mask_prev=None, mask_blend_weight=1.0):
        """Shared tail of compute_blended_bone_matrices and compute_
        blended_bone_matrices_multi: given each track's already-sampled
        pose (translations, rotations, scales), composites them per-joint
        by upper_joint_mask (upper_pose=None means every joint just uses
        lower_pose, matching a track with no upper clip configured/
        playing), applies upper_rotation_offsets, and walks the hierarchy
        via _world_matrices. See compute_blended_bone_matrices's own
        docstring for exactly what upper_rotation_offsets/upper_joint_
        mask_prev/mask_blend_weight do - this is purely the composition
        step, agnostic to how each pose was sampled."""
        lower_t, lower_r, lower_s = lower_pose
        if upper_pose is None:
            return self._world_matrices(lower_t, lower_r, lower_s)

        upper_t, upper_r, upper_s = upper_pose

        mask_transitioning = (
            upper_joint_mask_prev is not None and mask_blend_weight < 1.0
            and upper_joint_mask_prev != upper_joint_mask
        )
        if not mask_transitioning:
            translations = [upper_t[i] if upper_joint_mask[i] else lower_t[i] for i in range(len(self.joints))]
            rotations = [upper_r[i] if upper_joint_mask[i] else lower_r[i] for i in range(len(self.joints))]
            scales = [upper_s[i] if upper_joint_mask[i] else lower_s[i] for i in range(len(self.joints))]
        else:
            identity_translation = glm.vec3(0.0)
            identity_rotation = glm.quat(1.0, 0.0, 0.0, 0.0)
            identity_scale = glm.vec3(1.0)
            translations, rotations, scales = [], [], []
            for i in range(len(self.joints)):
                if upper_joint_mask[i] == upper_joint_mask_prev[i]:
                    translations.append(upper_t[i] if upper_joint_mask[i] else lower_t[i])
                    rotations.append(upper_r[i] if upper_joint_mask[i] else lower_r[i])
                    scales.append(upper_s[i] if upper_joint_mask[i] else lower_s[i])
                    continue
                # This joint's mask assignment just changed - slerp/mix
                # between its old and new pose source instead of a hard
                # cut. None (an unanimated channel) resolves to identity
                # here, matching _local_matrix's own None convention
                # elsewhere, NOT the joint's bind pose.
                old_t = upper_t[i] if upper_joint_mask_prev[i] else lower_t[i]
                new_t = upper_t[i] if upper_joint_mask[i] else lower_t[i]
                old_r = upper_r[i] if upper_joint_mask_prev[i] else lower_r[i]
                new_r = upper_r[i] if upper_joint_mask[i] else lower_r[i]
                old_s = upper_s[i] if upper_joint_mask_prev[i] else lower_s[i]
                new_s = upper_s[i] if upper_joint_mask[i] else lower_s[i]
                translations.append(glm.mix(
                    old_t if old_t is not None else identity_translation,
                    new_t if new_t is not None else identity_translation,
                    mask_blend_weight,
                ))
                rotations.append(glm.slerp(
                    old_r if old_r is not None else identity_rotation,
                    new_r if new_r is not None else identity_rotation,
                    mask_blend_weight,
                ))
                scales.append(glm.mix(
                    old_s if old_s is not None else identity_scale,
                    new_s if new_s is not None else identity_scale,
                    mask_blend_weight,
                ))

        return self._world_matrices(translations, rotations, scales, rotation_offsets=upper_rotation_offsets)


def _extract_material(ctx, gltf, blob, material_index, glb_dir):
    """Returns (base_color, metallic, roughness, emissive, texture,
    mr_texture) - texture/mr_texture are already-uploaded moderngl
    textures (or None if ctx is None, no material, or no texture on
    it), matching model_loader.py's _extract_material return shape so
    the resulting obj dict slots into bind_material() unmodified.

    base_color is the material's own baseColorFactor - used as the
    vertex-color fallback tint when there's no texture (same role
    model_loader.py's base_color plays), not just a hardcoded white."""
    base_color = np.array([0.8, 0.8, 0.8], dtype="f4")
    metallic, roughness = 1.0, 1.0
    emissive = np.zeros(3, dtype="f4")
    texture = None
    mr_texture = None

    if material_index is None:
        return base_color, metallic, roughness, emissive, texture, mr_texture

    mat = gltf["materials"][material_index]
    pbr = mat.get("pbrMetallicRoughness", {})

    if "baseColorFactor" in pbr:
        base_color = np.array(pbr["baseColorFactor"][:3], dtype="f4")

    metallic = float(pbr.get("metallicFactor", 1.0))
    roughness = float(pbr.get("roughnessFactor", 1.0))

    if "emissiveFactor" in mat:
        emissive = np.array(mat["emissiveFactor"], dtype="f4")

    def load_texture(tex_info):
        if tex_info is None or ctx is None:
            return None

        image_index = gltf["textures"][tex_info["index"]].get("source")
        if image_index is None:
            return None

        image_entry = gltf["images"][image_index]
        raw_bytes = None

        if "bufferView" in image_entry:
            view = gltf["bufferViews"][image_entry["bufferView"]]
            offset = view.get("byteOffset", 0)
            length = view["byteLength"]
            raw_bytes = blob[offset:offset + length]
        elif "uri" in image_entry and glb_dir is not None:
            try:
                with open(glb_dir / image_entry["uri"], "rb") as f:
                    raw_bytes = f.read()
            except Exception as e:
                print(f"[skeletal_loader] Failed to load external image '{image_entry['uri']}': {e}")
                return None

        if raw_bytes is None:
            return None

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        tex = ctx.texture(img.size, 3, img.tobytes())
        tex.build_mipmaps()
        # Set once here, not per-frame in pbr_shader.py's
        # _bind_material_textures (see that function's own comment on
        # why it no longer touches .filter at all) - matches
        # model_loader.py's _upload_texture, the other texture-creation
        # site feeding the same bind_material() path.
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = tex.repeat_y = True
        return tex

    texture = load_texture(pbr.get("baseColorTexture"))
    mr_texture = load_texture(pbr.get("metallicRoughnessTexture"))

    return base_color, metallic, roughness, emissive, texture, mr_texture


def _read_glb_json_and_blob(path):
    """Parses a .glb container's chunk table and returns (gltf, blob) -
    gltf the decoded JSON dict, blob the raw binary chunk (or None if
    the file has no BIN chunk). Returns None if the file has no JSON
    chunk at all. Shared by load_skinned_glb and load_animation_clips -
    both need the same raw container access, just different things out
    of the parsed JSON afterward."""
    with open(path, "rb") as f:
        data = f.read()

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"'{path}' is not a valid .glb file")

    json_chunk = None
    blob = None
    offset = 12
    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        chunk_data = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == b"JSON":
            json_chunk = chunk_data
        elif chunk_type == b"BIN\x00":
            blob = chunk_data
        offset += 8 + chunk_length

    if json_chunk is None:
        return None

    return json.loads(json_chunk.decode("utf-8")), blob


def _parse_animations(gltf, blob, node_to_joint_index, name_prefix="", time_scale=1.0):
    """Shared by load_skinned_glb and load_animation_clips - reads every
    animation in gltf["animations"], keeping only channels that target a
    node present in node_to_joint_index (a node->joint-index mapping;
    load_animation_clips passes one keyed by joint NAME matches against
    a different skeleton than the file's own, so a channel targeting a
    joint that skeleton doesn't have is silently dropped rather than
    crashing - see load_animation_clips). name_prefix is prepended to
    every clip's dict key (not its own .name attribute) only to keep
    load_animation_clips' caller-facing rename simple; pass "" (the
    load_skinned_glb case) for no change.

    time_scale: multiplies every keyframe TIME (seconds) by this factor
    after reading - see load_skinned_glb/load_animation_clips' own
    time_scale docstring for why this exists (correcting a source file
    actually baked at a different frame rate than intended). 1.0 (the
    default) is a no-op - glTF keyframe times are already real seconds
    by spec, correctly converted by a compliant exporter, so trusting
    them as-is is the right default; this is strictly an opt-in
    override, never applied automatically."""
    animations = {}
    for anim_index, anim in enumerate(gltf.get("animations", [])):
        name = anim.get("name", f"animation_{anim_index}")
        samplers = anim["samplers"]
        channels = []

        for chan in anim["channels"]:
            target = chan["target"]
            target_node = target.get("node")
            if target_node is None or target_node not in node_to_joint_index:
                continue  # targets something outside this skin - ignore

            joint_index = node_to_joint_index[target_node]
            path = target["path"]
            if path not in ("translation", "rotation", "scale"):
                continue  # e.g. "weights" (morph targets) - not supported here

            sampler = samplers[chan["sampler"]]
            times = _read_accessor(gltf, blob, sampler["input"])
            if time_scale != 1.0:
                times = times * time_scale
            values = _read_accessor(gltf, blob, sampler["output"])
            interpolation = sampler.get("interpolation", "LINEAR")

            if interpolation == "CUBICSPLINE":
                print(
                    f"[skeletal_loader] {path}: CUBICSPLINE interpolation on '{name}' "
                    f"joint {joint_index} is not supported - treating as LINEAR, "
                    f"which will look wrong (the accessor packs in-tangent/value/"
                    f"out-tangent triples, not plain values)."
                )
                # Best-effort: cubicspline output is 3x as many entries
                # (in-tangent, value, out-tangent per keyframe) - grab
                # just the value component so this doesn't crash, even
                # though the curve shape itself will be wrong.
                component_count = values.shape[1] if values.ndim > 1 else 1
                values = values.reshape(-1, 3, component_count)[:, 1, :]
                interpolation = "LINEAR"

            channels.append(AnimationChannel(joint_index, path, times, values, interpolation))

        if channels:
            animations[name_prefix + name] = AnimationClip(name_prefix + name, channels)

    return animations


def load_animation_clips(path, skeleton, rename=None, time_scale=1.0):
    """Loads animation clips from a SEPARATE .glb file and merges them
    directly into an already-loaded skeleton's own .animations dict (in
    place - nothing is returned besides the list of clip names actually
    added). For a rig exported as several files sharing one armature -
    e.g. a base character glb plus separate "pose" glbs each holding one
    baked animation for the same skeleton (this project's rat.glb plus
    Assets/Animations/Poses/Rifle/*.glb) - so a new pose file can be
    dropped in and played on the existing mesh without re-exporting or
    duplicating it.

    Unlike load_skinned_glb, this does NOT require exactly one skin/mesh
    or any particular scope constraints on the source file's geometry -
    the source file's own mesh/material data is ignored entirely, only
    its skin's joint list (for NAME-matching against `skeleton`, since
    two independently-exported files are not guaranteed to share the
    same node ordering even when they share the same joint names) and
    its animations are read.

    rename: optional dict mapping the source file's own clip name(s) to
    whatever name they should be stored under in `skeleton.animations`
    (e.g. {"New": "rifle_idle"} - both of this project's rifle pose
    files happen to name their one clip "New", the same name rat.glb's
    own base clip already uses, so they'd silently collide/overwrite
    without renaming). A clip name not present in `rename` keeps its
    original name from the source file.

    Returns the list of clip names actually added to `skeleton.
    animations` (after renaming) - empty if the file had no animations,
    or if none of its channels targeted a joint `skeleton` actually has.

    time_scale: see _parse_animations - an opt-in correction multiplier
    for every keyframe time, for when a specific source file is known
    to have been baked at the wrong frame rate (e.g. Blender's factory-
    default 24fps scene setting left unchanged when 30fps was actually
    intended - see the call sites currently using this for exactly that
    case). 1.0 (no change) unless a caller has a specific reason to
    believe THIS file's baked rate doesn't match what was intended."""
    result = _read_glb_json_and_blob(path)
    if result is None:
        print(f"[skeletal_loader] {path} has no JSON chunk - not a valid .glb, no animations loaded.")
        return []
    gltf, blob = result

    skins = gltf.get("skins", [])
    if not skins:
        print(f"[skeletal_loader] {path} has no skin - can't match its animation channels to any joint.")
        return []

    source_joint_node_indices = skins[0]["joints"]
    target_joint_index_by_name = {joint.name: i for i, joint in enumerate(skeleton.joints)}

    # Map each SOURCE node index straight to the TARGET skeleton's joint
    # index (by name) - this is the key difference from load_skinned_
    # glb's own node_to_joint_index, which maps a node to a position in
    # that same file's OWN joint list. A source joint with no matching
    # name in the target skeleton is left out of this mapping entirely,
    # so _parse_animations' existing "not in node_to_joint_index -
    # ignore" branch already does the right thing for it for free.
    node_to_target_joint_index = {}
    unmatched = []
    for source_node_idx in source_joint_node_indices:
        joint_name = gltf["nodes"][source_node_idx].get("name")
        if joint_name in target_joint_index_by_name:
            node_to_target_joint_index[source_node_idx] = target_joint_index_by_name[joint_name]
        else:
            unmatched.append(joint_name)

    if unmatched:
        print(
            f"[skeletal_loader] {path}: {len(unmatched)} joint(s) with no "
            f"name match in the target skeleton - any channels targeting "
            f"them are dropped: {unmatched}"
        )

    parsed = _parse_animations(gltf, blob, node_to_target_joint_index, time_scale=time_scale)

    rename = rename or {}
    added = []
    for source_name, clip in parsed.items():
        final_name = rename.get(source_name, source_name)
        if final_name in skeleton.animations:
            print(
                f"[skeletal_loader] {path}: clip '{final_name}' already "
                f"exists on this skeleton - overwriting with the one from "
                f"this file."
            )
        clip.name = final_name
        skeleton.animations[final_name] = clip
        added.append(final_name)

    return added


def load_hat(hat_source, name, ctx):
    """One hat's geometry (same skin/joint indexing as the base mesh, so it
    animates with it) plus its own texture, in the same dict shape
    create_skeletal_vao takes. None if the hat doesn't exist/can't be read."""
    prims = hat_source["prims"].get(name)
    if not prims:
        return None
    gltf, blob = hat_source["gltf"], hat_source["blob"]
    parts = {k: [] for k in ("positions", "normals", "uvs", "joints_0", "weights_0", "faces")}
    offset = 0
    for prim in prims:
        attrs = prim.get("attributes", {})
        if not all(k in attrs for k in ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0")):
            return None
        pos = _read_accessor(gltf, blob, attrs["POSITION"])
        parts["positions"].append(pos)
        parts["normals"].append(_read_accessor(gltf, blob, attrs["NORMAL"]))
        parts["uvs"].append(
            _read_accessor(gltf, blob, attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs
            else np.zeros((len(pos), 2), dtype=np.float32))
        parts["joints_0"].append(_read_accessor(gltf, blob, attrs["JOINTS_0"]).astype(np.uint32))
        parts["weights_0"].append(_read_accessor(gltf, blob, attrs["WEIGHTS_0"]).astype(np.float32))
        idx = prim.get("indices")
        faces = (_read_accessor(gltf, blob, idx).astype(np.uint32).reshape(-1, 3) if idx is not None
                 else np.arange(len(pos), dtype=np.uint32).reshape(-1, 3))
        parts["faces"].append(faces + offset)
        offset += len(pos)
    data = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    base_color, _m, _r, _e, texture, _mr = _extract_material(
        ctx, gltf, blob, prims[0].get("material"), hat_source["dir"])
    data["base_color"] = base_color
    data["texture"] = texture
    return data


def load_skinned_glb(path, ctx=None, time_scale=1.0):
    """Returns a dict:
        {
            "positions": (N, 3) float32,
            "normals": (N, 3) float32,
            "uvs": (N, 2) float32,
            "faces": (M, 3) uint32,
            "joints_0": (N, 4) uint32 - up to 4 joint indices per vertex,
            "weights_0": (N, 4) float32 - matching blend weights,
            "skeleton": Skeleton,
            "base_color": (3,) float32,
            "metallic": float,
            "roughness": float,
            "emissive": (3,) float32,
            "texture": moderngl.Texture or None,
            "metallic_roughness_texture": moderngl.Texture or None,
        }
    or None if the file has no skin (not a skeletal mesh) or doesn't
    meet the scope constraints documented at the top of this file.

    ctx: a moderngl context, used to upload textures found in the
    material. Pass None to skip texture upload entirely (still returns
    geometry/animation/material-factor data).

    time_scale: see _parse_animations/load_animation_clips' own
    docstring - an opt-in per-keyframe-time correction multiplier for a
    file known to have been baked at the wrong frame rate. 1.0 (no
    change) by default."""
    result = _read_glb_json_and_blob(path)
    if result is None:
        return None
    gltf, blob = result

    skins = gltf.get("skins", [])
    if not skins:
        print(f"[skeletal_loader] {path} has no skin - not a skeletal mesh.")
        return None
    skin = skins[0]
    if len(skins) > 1:
        print(f"[skeletal_loader] {path} has {len(skins)} skins - only using the first.")

    # Gather primitives via the NODES that reference each mesh, not by
    # flattening every mesh in the file unconditionally - a node's own
    # name is what _is_hidden_by_default matches against (see its
    # comment), and a mesh can only be excluded/included through
    # whichever node actually places it in the scene. A mesh referenced
    # by more than one node (not the case for rat.glb, but not assumed
    # impossible) contributes its primitives once per non-hidden node
    # that references it.
    meshes = gltf.get("meshes", [])
    prims = []
    hidden_node_names = []
    hat_prims = {}
    for node in gltf.get("nodes", []):
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        node_name = node.get("name", f"mesh_{mesh_index}")
        if _is_hidden_by_default(node_name):
            hidden_node_names.append(node_name)
            if node_name.lower().startswith(_HAT_PREFIX):
                hat_prims[node_name[len(_HAT_PREFIX):]] = list(meshes[mesh_index].get("primitives", []))
            continue
        prims.extend(meshes[mesh_index].get("primitives", []))

    if hidden_node_names:
        print(f"[skeletal_loader] {path}: hiding by default (name convention): {hidden_node_names}")

    if not prims:
        print(f"[skeletal_loader] {path} has no visible mesh primitives, aborting.")
        return None

    required = ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0")

    # Merge every primitive sharing this skin into one combined mesh -
    # a glTF node's skin applies to ALL of its mesh's primitives (skin is
    # a node-level reference, not per-primitive), and multiple primitives
    # per mesh is the normal, common result of exporting a model with more
    # than one material slot (e.g. body/eyes/teeth as separate slots) -
    # nothing unusual about the asset, just something this loader didn't
    # handle before. JOINTS_0 values are already absolute indices into
    # the shared skin's joint list, so they need no adjustment across
    # primitives - only face indices need offsetting, same as
    # model_loader.py's _flatten_scene does for its own vertex_offset.
    position_parts, normal_parts, uv_parts = [], [], []
    joints_parts, weights_parts, face_parts = [], [], []
    vertex_offset = 0
    material_indices = []

    for prim in prims:
        attrs = prim.get("attributes", {})
        if not all(name in attrs for name in required):
            print(f"[skeletal_loader] {path} is missing one of {required} on a primitive, aborting.")
            return None

        prim_positions = _read_accessor(gltf, blob, attrs["POSITION"])
        prim_normals = _read_accessor(gltf, blob, attrs["NORMAL"])
        prim_uvs = (
            _read_accessor(gltf, blob, attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs
            else np.zeros((len(prim_positions), 2), dtype=np.float32)
        )
        prim_joints = _read_accessor(gltf, blob, attrs["JOINTS_0"]).astype(np.uint32)
        prim_weights = _read_accessor(gltf, blob, attrs["WEIGHTS_0"]).astype(np.float32)

        indices_accessor = prim.get("indices")
        if indices_accessor is not None:
            prim_faces = _read_accessor(gltf, blob, indices_accessor).astype(np.uint32).reshape(-1, 3)
        else:
            prim_faces = np.arange(len(prim_positions), dtype=np.uint32).reshape(-1, 3)

        position_parts.append(prim_positions)
        normal_parts.append(prim_normals)
        uv_parts.append(prim_uvs)
        joints_parts.append(prim_joints)
        weights_parts.append(prim_weights)
        face_parts.append(prim_faces + vertex_offset)
        vertex_offset += len(prim_positions)
        material_indices.append(prim.get("material"))

    positions = np.concatenate(position_parts, axis=0)
    normals = np.concatenate(normal_parts, axis=0)
    uvs = np.concatenate(uv_parts, axis=0)
    joints_0 = np.concatenate(joints_parts, axis=0)
    weights_0 = np.concatenate(weights_parts, axis=0)
    faces = np.concatenate(face_parts, axis=0)

    if len(set(material_indices)) > 1:
        print(
            f"[skeletal_loader] {path} has {len(prims)} primitives using "
            f"{len(set(material_indices))} different materials - only the "
            f"first primitive's material (base color/metallic/roughness/"
            f"emissive/textures) is used for the WHOLE merged mesh. Same "
            f"simplification model_loader.py's own multi-object flatten "
            f"already makes for a non-skinned multi-material mesh - true "
            f"per-primitive materials would need per-primitive draw calls "
            f"(or a texture atlas), not implemented here."
        )

    # --- Skin / joint hierarchy ---
    joint_node_indices = skin["joints"]
    inv_bind_raw = _read_accessor(gltf, blob, skin["inverseBindMatrices"]) if "inverseBindMatrices" in skin else None

    parent_map = _build_parent_map(gltf)
    node_to_joint_index = {node_idx: i for i, node_idx in enumerate(joint_node_indices)}

    joints = []
    for i, node_idx in enumerate(joint_node_indices):
        node = gltf["nodes"][node_idx]
        parent_node_idx = parent_map.get(node_idx, -1)
        parent_joint_idx = node_to_joint_index.get(parent_node_idx, -1)

        local_bind = _node_local_matrix(node)

        if inv_bind_raw is not None:
            inv_bind = glm.mat4(*inv_bind_raw[i])
        else:
            inv_bind = glm.mat4(1.0)

        # parent_joint_idx == -1 here can mean either "genuinely no
        # parent" OR "has a parent, but it's outside the skin's own
        # joint list" (e.g. an armature object node) - only the second
        # case needs an external_root_matrix; a truly parentless node's
        # ancestor walk just naturally comes back identity anyway, but
        # computing it unconditionally for every -1 case is simpler than
        # re-deriving that distinction here, and free (this only runs
        # once at load time, not per frame).
        external_root_matrix = (
            _external_ancestor_matrix(node_idx, parent_map, gltf) if parent_joint_idx == -1 else None
        )

        joints.append(Joint(
            name=node.get("name", f"joint_{i}"),
            node_index=node_idx,
            parent_joint_index=parent_joint_idx,
            local_bind_matrix=local_bind,
            inverse_bind_matrix=inv_bind,
            external_root_matrix=external_root_matrix,
        ))

    # --- Animations ---
    animations = _parse_animations(gltf, blob, node_to_joint_index, time_scale=time_scale)
    skeleton = Skeleton(joints, animations)

    material_index = material_indices[0]
    base_color, metallic, roughness, emissive, texture, mr_texture = _extract_material(
        ctx, gltf, blob, material_index, Path(path).parent
    )

    return {
        "positions": positions,
        "normals": normals,
        "uvs": uvs,
        "faces": faces,
        "joints_0": joints_0,
        "weights_0": weights_0,
        "skeleton": skeleton,
        "base_color": base_color,
        "metallic": metallic,
        "roughness": roughness,
        "emissive": emissive,
        "texture": texture,
        "metallic_roughness_texture": mr_texture,
        # Only what load_hat needs later; nothing is parsed/uploaded until a
        # hat is actually asked for.
        "hat_source": {"gltf": gltf, "blob": blob, "dir": Path(path).parent, "prims": hat_prims},
    }


def create_skeletal_vao(ctx, prog, skinned_data):
    """Builds VBOs + a VAO bound to prog from load_skinned_glb()'s
    output. Only creates a VBO for an attribute if prog actually
    declares it (same pattern as model_loader.py's _has_attribute) -
    so the same skinned_data can build both the full render VAO (needs
    in_uv/in_color/in_lightmap_uv) and the lean skinned shadow VAO
    (only needs in_position/in_joints/in_weights) against their
    respective programs without creating unused buffers.

    NOTE: in_joints is uploaded as signed 32-bit ints ("4i") to match
    GLSL's ivec4. moderngl is expected to introspect the shader's
    declared attribute type and use the integer vertex-attribute path
    (glVertexAttribIPointer) rather than the float path automatically -
    this is the standard/documented approach, but hasn't been verified
    against a real GPU in this environment (no OpenGL context available
    here), unlike most other rendering code in this project which was
    compile/run-tested directly.

    Returns {"vao": ..., "vbos": {...}, "ibo": ...} - caller (Scene)
    should hang onto "vbos" and "ibo" for cleanup in destroy(), same as
    every other object dict already does for "vao"/"shadow_vao"."""
    positions = skinned_data["positions"].astype("f4")
    normals = skinned_data["normals"].astype("f4")
    uvs = skinned_data["uvs"].astype("f4")
    faces = skinned_data["faces"].astype("i4")
    joints_0 = skinned_data["joints_0"].astype("i4")
    weights_0 = skinned_data["weights_0"].astype("f4")
    colors = np.tile(skinned_data.get("base_color", np.array([1.0, 1.0, 1.0], dtype="f4")), (len(positions), 1)).astype("f4")
    lightmap_uvs = np.zeros((len(positions), 2), dtype="f4")

    attr_data = {
        "in_position": (positions, "3f"),
        "in_normal": (normals, "3f"),
        "in_color": (colors, "3f"),
        "in_uv": (uvs, "2f"),
        "in_lightmap_uv": (lightmap_uvs, "2f"),
        "in_joints": (joints_0, "4i"),
        "in_weights": (weights_0, "4f"),
    }

    def has_attr(name):
        try:
            return prog[name] is not None
        except Exception:
            return False

    vbos = {name: ctx.buffer(data.tobytes()) for name, (data, fmt) in attr_data.items() if has_attr(name)}
    ibo = ctx.buffer(faces.tobytes())

    vao_content = [(vbos[name], attr_data[name][1], name) for name in vbos]
    vao = ctx.vertex_array(prog, vao_content, ibo)

    return {"vao": vao, "vbos": vbos, "ibo": ibo}