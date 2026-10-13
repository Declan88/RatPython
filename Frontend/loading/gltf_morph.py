"""
Loads a glb file's morph-target ("shape key") geometry directly from its
raw glTF JSON+BIN chunks - completely independent of model_loader.py's
trimesh-based pipeline, which has NO concept of morph targets at all
(trimesh.load just doesn't expose glTF "targets" data), so there's
nothing in that pipeline to extend; this bypasses it entirely instead,
the same way gltf_lights.py/gltf_actors.py bypass it for lights/actor
markers.

Also hand-rolls its OWN minimal per-primitive material (base color
texture + metallic/roughness factors only - no normal/emissive maps,
which none of this project's morph-bearing models use yet) rather than
reusing load_glb_by_material: that function's own per-primitive vertex
order isn't guaranteed to line up with the morph target deltas read here
(both are read independently, by different code, from the same file) -
reading EVERY attribute (base position/normal/uv/indices AND the morph
deltas) from the exact same raw accessors, in this one place, is what
actually guarantees per-vertex correspondence between a target's delta
and the base vertex it offsets, by construction rather than by hoping
two separate loaders agree.

Call load_morph_primitives(glb_path) - see its own docstring for the
returned shape. Scene.add_dynamic_morph (scene_base.py) is the only
caller; it uploads everything this returns as VBOs/a texture and builds
a VAO against self.morph_program (pbr_shader.py's own morph-capable
variant of the ordinary static/dynamic vertex shader).
"""

import io
import json
import struct

import numpy as np
from PIL import Image

_COMPONENT_FORMATS = {
    5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
    5125: ("I", 4), 5126: ("f", 4),
}
_TYPE_COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _read_chunks(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"'{path}' is not a valid .glb file")
    json_chunk = None
    bin_chunk = None
    offset = 12
    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        chunk_data = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == b"JSON":
            json_chunk = chunk_data
        elif chunk_type[:3] == b"BIN":
            bin_chunk = chunk_data
        offset += 8 + chunk_length
    return json_chunk, bin_chunk


def _read_accessor(gltf, bin_data, accessor_index):
    """Returns a (count, components) float32 array for accessor_index -
    every attribute this file actually reads (POSITION/NORMAL/TEXCOORD_0,
    their morph-target counterparts, and indices) is a plain, non-sparse
    accessor in every model this has been used on; a sparse accessor
    raises instead of silently returning the un-overridden base data."""
    accessor = gltf["accessors"][accessor_index]
    if accessor.get("sparse") is not None:
        raise ValueError(f"accessor {accessor_index}: sparse accessors aren't supported here")
    count = accessor["count"]
    components = _TYPE_COMPONENT_COUNTS[accessor["type"]]
    fmt_char, comp_size = _COMPONENT_FORMATS[accessor["componentType"]]
    buffer_view = gltf["bufferViews"][accessor["bufferView"]]
    byte_offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = buffer_view.get("byteStride") or comp_size * components
    out = np.empty((count, components), dtype=np.float32)
    for i in range(count):
        base = byte_offset + i * stride
        out[i] = struct.unpack_from(f"<{components}{fmt_char}", bin_data, base)
    return out


def _node_matrices_by_mesh(gltf):
    """Maps mesh_index -> (model_matrix (3,3) + translation (3,), normal_
    matrix (3,3)) for the node that references it - a primitive's own
    POSITION/NORMAL/target-delta accessors are in MESH-LOCAL space
    (exactly like model_loader.py's trimesh path, which DOES apply each
    node's own transform when flattening the scene - trimesh just has no
    visibility into morph targets, not into node transforms), so skipping
    this step (an earlier version of this file did) renders every
    morph-bearing object at identity scale/rotation/translation instead
    of whatever its own node in the .glb actually places it at - exactly
    what made Cheesepile.glb's "Cheese" node (scale [4, 0.2, 4],
    translation [0, -2.07, 0] - confirmed via direct inspection) render
    5x too tall (missing that 0.2 Y scale) with equally 5x-oversized
    morph delta amplitude riding on top of it. Assumes one node per mesh
    (true for every morph-bearing model so far) - a mesh referenced by
    more than one node would need per-INSTANCE primitives instead of
    per-mesh, not supported here."""
    result = {}
    for node in gltf.get("nodes", []):
        mesh_index = node.get("mesh")
        if mesh_index is None or mesh_index in result:
            continue
        if "matrix" in node:
            # glTF matrices are column-major - reshape(4,4) then transpose
            # to get a row-major numpy array indexable the usual [row,col] way.
            m = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
            upper3, translation = m[:3, :3], m[:3, 3]
        else:
            x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
            rot = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64)
            sx, sy, sz = node.get("scale", [1.0, 1.0, 1.0])
            upper3 = rot @ np.diag([sx, sy, sz])
            translation = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
        # Correct normal transform under non-uniform scale is the inverse-
        # transpose of the position matrix's own upper3x3 - for an
        # orthogonal rotation times a diagonal scale (no shear, true of
        # every TRS-authored node here), that's just rot @ diag(1/s) - the
        # rotation part is self-inverse-transpose, only the scale part
        # needs inverting.
        normal_matrix = np.linalg.inv(upper3).T
        result[mesh_index] = (upper3.astype(np.float32), translation.astype(np.float32),
                               normal_matrix.astype(np.float32))
    return result


def _read_image(gltf, bin_data, texture_ref):
    if texture_ref is None:
        return None
    texture = gltf["textures"][texture_ref["index"]]
    image = gltf["images"][texture["source"]]
    buffer_view = gltf["bufferViews"][image["bufferView"]]
    offset = buffer_view.get("byteOffset", 0)
    length = buffer_view["byteLength"]
    return Image.open(io.BytesIO(bin_data[offset:offset + length]))


def load_morph_primitives(glb_path):
    """Returns a list of per-primitive dicts (one per mesh primitive in
    the WHOLE file, in file order - Cheesepile.glb has two: "Cheese",
    which owns the morph target, and "Plate", which doesn't), or None if
    the file has no morph target anywhere in it at all (nothing calls
    this for a file that doesn't need it, but it's a clean signal if one
    ever does).

    Each dict: {
        "material_name": str or None,
        "position"/"normal": (N, 3) float32,
        "uv": (N, 2) float32,
        "indices": (M,) uint32,
        "morph_delta_position"/"morph_delta_normal": (N, 3) float32 -
            all zero for a primitive with no morph target of its own (so
            it can still be drawn through the same morph-aware VAO/
            program as a sibling primitive that DOES have one, at zero
            cost - see Scene.add_dynamic_morph), with "target_names": []
            in that case.
        "target_names": [str, ...] - this project only ever authors (and
            Scene.add_dynamic_morph only ever drives) ONE target per
            primitive right now, so only index 0's delta is read even if
            a primitive somehow had more.
        "base_color_factor": (r, g, b, a) 0-1,
        "metallic"/"roughness": float,
        "double_sided": bool,
        "image": PIL.Image or None - the base color texture, undecoded
            beyond PIL's own lazy load (Scene uploads it).
    }"""
    json_chunk, bin_chunk = _read_chunks(glb_path)
    if json_chunk is None or bin_chunk is None:
        return None
    gltf = json.loads(json_chunk.decode("utf-8"))
    materials = gltf.get("materials", [])

    meshes = gltf.get("meshes", [])
    if not any(p.get("targets") for m in meshes for p in m.get("primitives", [])):
        return None

    node_matrices = _node_matrices_by_mesh(gltf)

    results = []
    for mesh_index, mesh in enumerate(meshes):
        upper3, translation, normal_matrix = node_matrices.get(
            mesh_index, (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
        )
        for primitive in mesh.get("primitives", []):
            attrs = primitive["attributes"]
            position = _read_accessor(gltf, bin_chunk, attrs["POSITION"])
            normal = _read_accessor(gltf, bin_chunk, attrs["NORMAL"])
            uv = (
                _read_accessor(gltf, bin_chunk, attrs["TEXCOORD_0"])
                if "TEXCOORD_0" in attrs else np.zeros((len(position), 2), dtype=np.float32)
            )
            indices_accessor = primitive.get("indices")
            if indices_accessor is not None:
                indices = _read_accessor(gltf, bin_chunk, indices_accessor)[:, 0].astype(np.uint32)
            else:
                indices = np.arange(len(position), dtype=np.uint32)

            targets = primitive.get("targets") or []
            target_names = mesh.get("extras", {}).get("targetNames") or []
            if targets:
                target = targets[0]
                pos_idx = target.get("POSITION")
                morph_delta_position = (
                    _read_accessor(gltf, bin_chunk, pos_idx) if pos_idx is not None
                    else np.zeros_like(position)
                )
                norm_idx = target.get("NORMAL")
                morph_delta_normal = (
                    _read_accessor(gltf, bin_chunk, norm_idx) if norm_idx is not None
                    else np.zeros_like(normal)
                )
            else:
                morph_delta_position = np.zeros_like(position)
                morph_delta_normal = np.zeros_like(normal)

            # Bake this primitive's owning node's own TRS into every
            # position/normal/delta array - see _node_matrices_by_mesh's
            # own docstring for why this has to happen at all. Deltas get
            # the LINEAR part only (upper3/normal_matrix, no translation
            # - a delta is a direction/offset, not a point, so it
            # shouldn't be translated, only rotated/scaled the same way
            # the positions it's added to already are).
            position = position @ upper3.T + translation
            normal = normal @ normal_matrix.T
            morph_delta_position = morph_delta_position @ upper3.T
            morph_delta_normal = morph_delta_normal @ normal_matrix.T

            material_index = primitive.get("material")
            material = materials[material_index] if material_index is not None else {}
            pbr = material.get("pbrMetallicRoughness", {})

            results.append({
                "material_name": material.get("name"),
                "position": position,
                "normal": normal,
                "uv": uv,
                "indices": indices,
                "morph_delta_position": morph_delta_position,
                "morph_delta_normal": morph_delta_normal,
                "target_names": target_names if targets else [],
                "base_color_factor": tuple(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])),
                "metallic": float(pbr.get("metallicFactor", 1.0)),
                "roughness": float(pbr.get("roughnessFactor", 1.0)),
                "double_sided": bool(material.get("doubleSided", False)),
                "image": _read_image(gltf, bin_chunk, pbr.get("baseColorTexture")),
            })

    print(f"[gltf_morph] Loaded {len(results)} morph-aware primitive(s) from {glb_path}")
    return results
