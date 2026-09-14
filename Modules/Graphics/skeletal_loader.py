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
        self.times = times  # 1D numpy array, seconds
        self.values = values  # (N, 3) for translation/scale, (N, 4) xyzw for rotation
        self.interpolation = interpolation  # "LINEAR" | "STEP" (CUBICSPLINE falls back to LINEAR with a warning)

    def sample(self, time):
        times = self.times
        if len(times) == 0:
            return None
        if time <= times[0]:
            idx = 0
            t = 0.0
        elif time >= times[-1]:
            idx = len(times) - 2 if len(times) > 1 else 0
            t = 1.0 if len(times) > 1 else 0.0
        else:
            idx = int(np.searchsorted(times, time, side="right") - 1)
            idx = max(0, min(idx, len(times) - 2))
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
            value = v0 + (v1 - v0) * t
            return (self.path, glm.vec3(*value))


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


class Skeleton:
    def __init__(self, joints, animations):
        self.joints = joints  # list[Joint], in skin.joints order
        self.animations = animations  # dict[name -> AnimationClip]

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
        and animation_name at time), builds each joint's concrete local
        matrix via _local_matrix (bind-pose fallback already resolved,
        so there's always a real matrix on both sides even if one clip
        is None/missing a channel for this joint), decomposes each via
        glm.decompose, and interpolates the decomposed TRS components
        (translation/scale lerp, rotation slerp - matrices themselves
        can't be linearly interpolated, hence decomposing first)
        weighted by blend_weight (0 = entirely the outgoing pose, 1 =
        entirely the incoming one). This is what turns a hard instant
        cut between clips (the previous behavior - visible as a pop at
        every idle/walk state change) into a smooth transition."""
        cur_t, cur_r, cur_s = self._sample_clip(animation_name, time)
        if prev_animation_name is None or blend_weight >= 1.0:
            return cur_t, cur_r, cur_s

        prev_t, prev_r, prev_s = self._sample_clip(prev_animation_name, prev_time)

        joint_count = len(self.joints)
        translations, rotations, scales = [None] * joint_count, [None] * joint_count, [None] * joint_count
        blend_scale, blend_rot, blend_trans = glm.vec3(), glm.quat(), glm.vec3()
        blend_skew, blend_persp = glm.vec3(), glm.vec4()

        for i, joint in enumerate(self.joints):
            from_matrix = self._local_matrix(joint, prev_t[i], prev_r[i], prev_s[i])
            to_matrix = self._local_matrix(joint, cur_t[i], cur_r[i], cur_s[i])

            from_scale, from_rot, from_trans = glm.vec3(), glm.quat(), glm.vec3()
            glm.decompose(from_matrix, from_scale, from_rot, from_trans, glm.vec3(), glm.vec4())
            to_scale, to_rot, to_trans = glm.vec3(), glm.quat(), glm.vec3()
            glm.decompose(to_matrix, to_scale, to_rot, to_trans, glm.vec3(), glm.vec4())

            translations[i] = glm.mix(from_trans, to_trans, blend_weight)
            rotations[i] = glm.slerp(from_rot, to_rot, blend_weight)
            scales[i] = glm.mix(from_scale, to_scale, blend_weight)

        return translations, rotations, scales

    def _world_matrices(self, translations, rotations, scales):
        """Shared hierarchy walk: given a per-joint local pose (any
        entry may be None, meaning "use this joint's bind-pose local
        transform instead" - see AnimationClip.sample_pose), returns the
        final list of glm.mat4 skinning matrices (world * inverse_bind),
        one per joint, ready to upload as the GPU skinning palette. Used
        by both compute_bone_matrices (single clip) and
        compute_blended_bone_matrices (two clips, picked per joint by a
        mask) - the walk itself doesn't care where each joint's local
        pose came from."""
        world_cache = {}

        def world_matrix(i):
            if i in world_cache:
                return world_cache[i]

            joint = self.joints[i]
            local = self._local_matrix(joint, translations[i], rotations[i], scales[i])

            if joint.parent_joint_index == -1:
                # external_root_matrix is identity unless this joint's
                # real glTF parent lies outside the skin (e.g. an
                # armature object node with its own scale/translation/
                # rotation) - see Joint.external_root_matrix.
                world = joint.external_root_matrix * local
            else:
                world = world_matrix(joint.parent_joint_index) * local

            world_cache[i] = world
            return world

        return [world_matrix(i) * self.joints[i].inverse_bind_matrix for i in range(len(self.joints))]

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
        name_to_index = {joint.name: i for i, joint in enumerate(self.joints)}
        root_indices = {name_to_index[name] for name in root_joint_names if name in name_to_index}

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
                                       upper_prev_animation=None, upper_prev_time=0.0, upper_blend_weight=1.0):
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
        same as compute_bone_matrices' own defaults)."""
        lower_t, lower_r, lower_s = self._sample_track(
            lower_animation, lower_time, lower_prev_animation, lower_prev_time, lower_blend_weight
        )
        if upper_animation is None:
            return self._world_matrices(lower_t, lower_r, lower_s)

        upper_t, upper_r, upper_s = self._sample_track(
            upper_animation, upper_time, upper_prev_animation, upper_prev_time, upper_blend_weight
        )

        translations = [upper_t[i] if upper_joint_mask[i] else lower_t[i] for i in range(len(self.joints))]
        rotations = [upper_r[i] if upper_joint_mask[i] else lower_r[i] for i in range(len(self.joints))]
        scales = [upper_s[i] if upper_joint_mask[i] else lower_s[i] for i in range(len(self.joints))]
        return self._world_matrices(translations, rotations, scales)


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
    for node in gltf.get("nodes", []):
        mesh_index = node.get("mesh")
        if mesh_index is None:
            continue
        node_name = node.get("name", f"mesh_{mesh_index}")
        if _is_hidden_by_default(node_name):
            hidden_node_names.append(node_name)
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