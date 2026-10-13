"""
Extracts "actor" spawn markers from a .glb file, independent of whatever
model_loader.py uses to load real geometry - same raw glb-chunk/JSON
parsing and node-tree-walk-accumulating-world-matrices approach as
gltf_lights.py's own extract_punctual_lights (copied rather than shared,
matching this project's existing style - the same chunk-parsing logic is
already duplicated between gltf_lights.py and skeletal_loader.py's own
_read_glb_json_and_blob).

An actor marker is just an EMPTY node (no mesh, no light - authored as a
plain Empty in Blender) whose name is "actor_<id>", optionally with a
Blender-style ".001"/".002" duplicate-name suffix for placing more than one
of the same actor type in one map (Blender appends that automatically the
moment a second object shares a name - the suffix is stripped here so both
resolve to the same actor id, one spawned instance each, at their own
distinct transforms). The marker's own world transform (position/rotation/
scale - rotation in RADIANS, matching both Scene.add_static's own
`rotation=` param and PhysicsWorld's, so a caller never needs to convert)
is where Modules/Actors/registry.create_actor spawns that actor's own
dedicated model/collision - the marker itself carries no geometry of its
own and needs none.
"""

import json
import re
import struct

import glm

_ACTOR_NODE_RE = re.compile(r"^actor_([a-z0-9_]+)(?:\.\d+)?$", re.IGNORECASE)


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


def _collect_actor_markers(gltf):
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

        match = _ACTOR_NODE_RE.match(node.get("name", ""))
        if match is not None:
            results.append((match.group(1).lower(), world))

        for child_index in node.get("children", []):
            walk(child_index, world)

    for idx in root_indices:
        walk(idx, glm.mat4(1.0))

    return results


def extract_actor_markers(glb_path):
    """
    Returns a list of dicts, one per "actor_<id>"-named node found in the
    file:
        {
            "actor_id": "cheesepile",   # lower-cased, duplicate-suffix stripped
            "position": glm.vec3,       # world-space
            "rotation": glm.vec3,       # world-space, XYZ radians (see module docstring)
            "scale": glm.vec3,          # world-space
        }

    Returns [] if the file has no node whose name matches "actor_<id>" at
    all (the overwhelmingly common case - most maps place no actors).
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

    results = []
    for actor_id, world in _collect_actor_markers(gltf):
        scale, rot_quat, translation = glm.vec3(), glm.quat(), glm.vec3()
        skew, persp = glm.vec3(), glm.vec4()
        glm.decompose(world, scale, rot_quat, translation, skew, persp)
        results.append({
            "actor_id": actor_id,
            "position": translation,
            "rotation": glm.eulerAngles(rot_quat),
            "scale": scale,
        })

    if results:
        print(f"[gltf_actors] Found {len(results)} actor marker(s) in {glb_path}")

    return results
