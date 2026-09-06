"""
Extracts KHR_lights_punctual lights from a .glb file, independent of
whatever model_loader.py uses to load geometry. Uses only stdlib (json,
struct) plus glm for the node-hierarchy math, so it doesn't need to know
anything about how the rest of your pipeline parses glb.

glTF puts lights in two places:
  - gltf["extensions"]["KHR_lights_punctual"]["lights"]: the light
    definitions (type, color, intensity, range).
  - a node's "extensions"."KHR_lights_punctual"."light" field: an index
    into the list above, meaning "this node has this light attached".
    The light's position/orientation comes from that node's transform,
    which can be nested arbitrarily deep in the scene graph - so this
    walks the full node tree accumulating world matrices rather than
    assuming lights sit at the top level.
"""

import json
import struct
import glm


def _node_local_matrix(node):
    if "matrix" in node:
        m = node["matrix"]
        return glm.mat4(*m)

    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # gltf order: x, y, z, w
    scale = node.get("scale", [1.0, 1.0, 1.0])

    t = glm.translate(glm.mat4(1.0), glm.vec3(*translation))
    q = glm.quat(rotation[3], rotation[0], rotation[1], rotation[2])  # glm order: w, x, y, z
    r = glm.mat4_cast(q)
    s = glm.scale(glm.mat4(1.0), glm.vec3(*scale))

    return t * r * s


def _collect_punctual_lights(gltf, light_defs):
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    scene_index = gltf.get("scene", 0)

    if scenes:
        root_indices = scenes[scene_index].get("nodes", [])
    else:
        # No explicit scene list - fall back to treating every node
        # that's never referenced as a child as a root.
        referenced = {c for n in nodes for c in n.get("children", [])}
        root_indices = [i for i in range(len(nodes)) if i not in referenced]

    results = []

    def walk(node_index, parent_matrix):
        node = nodes[node_index]
        world = parent_matrix * _node_local_matrix(node)

        light_ref = node.get("extensions", {}).get("KHR_lights_punctual")
        if light_ref is not None:
            light_def = light_defs[light_ref["light"]]
            position = glm.vec3(world[3])
            results.append((light_def, position))

        for child_index in node.get("children", []):
            walk(child_index, world)

    for idx in root_indices:
        walk(idx, glm.mat4(1.0))

    return results


def extract_punctual_lights(glb_path):
    """
    Returns a list of dicts, one per KHR_lights_punctual light found in
    the file (of any type - "point", "spot", or "directional"):
        {
            "type": "point" | "spot" | "directional",
            "position": glm.vec3,       # world-space, from the node transform
            "color": (r, g, b),         # 0..1, as authored
            "intensity": float,         # converted from candela/lux, see note below
            "range": float or None,     # meters; None means "infinite" per spec
        }

    Returns [] if the file has no KHR_lights_punctual extension at all
    (most glb files won't - this is only written when lights were placed
    in the authoring tool before export, e.g. Blender's "Punctual Lights"
    export option).

    Note on intensity units: glTF specifies point/spot intensity in
    candela and directional intensity in lux - physically-based units
    that don't map 1:1 onto an arbitrary shader's linear-light scale.
    This divides by 683 (the luminous-efficacy constant the Khronos
    sample viewer also uses for this conversion) to land in a roughly
    sane ballpark, but it's approximate - expect to retune per-scene.
    """
    with open(glb_path, "rb") as f:
        data = f.read()

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"'{glb_path}' is not a valid .glb file")

    json_chunk = None
    offset = 12
    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        chunk_data = data[offset + 8: offset + 8 + chunk_length]
        if chunk_type == b"JSON":
            json_chunk = chunk_data
            break
        offset += 8 + chunk_length

    if json_chunk is None:
        return []

    gltf = json.loads(json_chunk.decode("utf-8"))

    light_defs = (
        gltf.get("extensions", {})
        .get("KHR_lights_punctual", {})
        .get("lights", [])
    )
    if not light_defs:
        print(f"[gltf_lights] No punctual lights found in {glb_path}")
        return []

    results = []
    for light_def, position in _collect_punctual_lights(gltf, light_defs):
        raw_intensity = light_def.get("intensity", 1.0)

        results.append({
            "type": light_def.get("type", "point"),
            "position": position,
            "color": tuple(light_def.get("color", [1.0, 1.0, 1.0])),
            "intensity": raw_intensity / 683.0,
            "range": light_def.get("range"),
        })

    if results:
        print(f"[gltf_lights] Found {len(results)} punctual light(s) in {glb_path}")
    else:
        print(f"[gltf_lights] No punctual lights found in {glb_path}")

    return results