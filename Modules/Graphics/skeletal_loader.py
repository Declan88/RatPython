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
- One skin per file, one mesh primitive (same constraint model_loader.py
  already has for lightmap UVs - matching trimesh's flattened geometry
  order against pygltflib's raw structure isn't reliable for more than
  one primitive).
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


class Joint:
    __slots__ = ("name", "node_index", "parent_joint_index", "local_bind_matrix", "inverse_bind_matrix")

    def __init__(self, name, node_index, parent_joint_index, local_bind_matrix, inverse_bind_matrix):
        self.name = name
        self.node_index = node_index
        self.parent_joint_index = parent_joint_index  # index into Skeleton.joints, or -1
        self.local_bind_matrix = local_bind_matrix
        self.inverse_bind_matrix = inverse_bind_matrix


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

    def compute_bone_matrices(self, animation_name, time):
        """Returns a list of glm.mat4, one per joint (same order as
        self.joints), ready to upload as the GPU skinning palette."""
        clip = self.animations.get(animation_name)
        joint_count = len(self.joints)

        if clip is not None:
            translations, rotations, scales = clip.sample_pose(joint_count, time)
        else:
            translations = [None] * joint_count
            rotations = [None] * joint_count
            scales = [None] * joint_count

        world_cache = {}

        def world_matrix(i):
            if i in world_cache:
                return world_cache[i]

            joint = self.joints[i]

            if translations[i] is not None or rotations[i] is not None or scales[i] is not None:
                t = translations[i] if translations[i] is not None else glm.vec3(0.0)
                r = rotations[i] if rotations[i] is not None else glm.quat(1.0, 0.0, 0.0, 0.0)
                s = scales[i] if scales[i] is not None else glm.vec3(1.0)
                local = glm.translate(glm.mat4(1.0), t) * glm.mat4_cast(r) * glm.scale(glm.mat4(1.0), s)
            else:
                local = joint.local_bind_matrix

            if joint.parent_joint_index == -1:
                world = local
            else:
                world = world_matrix(joint.parent_joint_index) * local

            world_cache[i] = world
            return world

        return [world_matrix(i) * self.joints[i].inverse_bind_matrix for i in range(joint_count)]


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


def load_skinned_glb(path, ctx=None):
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
    geometry/animation/material-factor data)."""
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

    gltf = json.loads(json_chunk.decode("utf-8"))

    skins = gltf.get("skins", [])
    if not skins:
        print(f"[skeletal_loader] {path} has no skin - not a skeletal mesh.")
        return None
    skin = skins[0]
    if len(skins) > 1:
        print(f"[skeletal_loader] {path} has {len(skins)} skins - only using the first.")

    meshes = gltf.get("meshes", [])
    prims = [p for m in meshes for p in m.get("primitives", [])]
    if len(prims) != 1:
        print(f"[skeletal_loader] {path} has {len(prims)} primitive(s) - only exactly 1 is supported, aborting.")
        return None
    prim = prims[0]
    attrs = prim.get("attributes", {})

    required = ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0")
    if not all(name in attrs for name in required):
        print(f"[skeletal_loader] {path} is missing one of {required} on its primitive, aborting.")
        return None

    positions = _read_accessor(gltf, blob, attrs["POSITION"])
    normals = _read_accessor(gltf, blob, attrs["NORMAL"])
    uvs = _read_accessor(gltf, blob, attrs["TEXCOORD_0"]) if "TEXCOORD_0" in attrs else np.zeros((len(positions), 2), dtype=np.float32)
    joints_0 = _read_accessor(gltf, blob, attrs["JOINTS_0"]).astype(np.uint32)
    weights_0 = _read_accessor(gltf, blob, attrs["WEIGHTS_0"]).astype(np.float32)

    indices_accessor = prim.get("indices")
    if indices_accessor is not None:
        faces = _read_accessor(gltf, blob, indices_accessor).astype(np.uint32).reshape(-1, 3)
    else:
        faces = np.arange(len(positions), dtype=np.uint32).reshape(-1, 3)

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

        joints.append(Joint(
            name=node.get("name", f"joint_{i}"),
            node_index=node_idx,
            parent_joint_index=parent_joint_idx,
            local_bind_matrix=local_bind,
            inverse_bind_matrix=inv_bind,
        ))

    # --- Animations ---
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
            animations[name] = AnimationClip(name, channels)

    skeleton = Skeleton(joints, animations)

    material_index = prim.get("material")
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