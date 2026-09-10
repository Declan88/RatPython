from pathlib import Path
import moderngl
import numpy as np
from PIL import Image
import trimesh

DEFAULT_CREASE_ANGLE_DEG = 30.0

# An object named (or prefixed) this way in the authoring tool (e.g.
# Blender's outliner) never gets drawn - see _flatten_scene. It still
# comes through untouched on the PhysicsWorld.add_static_mesh side
# (physics_world.py's _load_mesh loads the whole glTF unfiltered), so
# naming a piece of geometry this way turns it into exactly the
# invisible-but-solid "clip brush" Source-family mapping uses for things
# like a smooth ramp collider over decorative stairs - model it as its
# own object in the same file, prefix its name, done. Matched against
# the object's name (what you rename in Blender's outliner), not the
# mesh data-block name, case-insensitively.
COLLISION_ONLY_PREFIX = "collision_"


def _is_collision_only_node(node_name):
    return node_name.lower().startswith(COLLISION_ONLY_PREFIX)

def _has_attribute(prog, name):
    try: return prog[name] is not None
    except Exception: return False

class _UnionFind:
    __slots__ = ("parent", "rank")
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    def find(self, x):
        root = x
        while self.parent[root] != root: root = self.parent[root]
        while self.parent[x] != root: self.parent[x], x = root, self.parent[x]
        return root
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1

def _canonical_position_ids(vertices, tolerance=1e-4):
    quantized = np.round(vertices / tolerance).astype(np.int64)
    _, pos_ids = np.unique(quantized, axis=0, return_inverse=True)
    return pos_ids

def _compute_vertex_normals(vertices, faces, crease_angle_deg=DEFAULT_CREASE_ANGLE_DEG):
    face_count = len(faces)
    if face_count == 0: return np.zeros_like(vertices, dtype="f4")
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    raw_face_normals = np.cross(v1 - v0, v2 - v0)
    face_lengths = np.linalg.norm(raw_face_normals, axis=1, keepdims=True)
    unit_face_normals = raw_face_normals / np.where(face_lengths < 1e-12, 1.0, face_lengths)
    pos_ids = _canonical_position_ids(vertices)
    face_pos = pos_ids[faces]

    edge_faces = {}
    for f in range(face_count):
        a, b, c = face_pos[f]
        for x, y in ((a, b), (b, c), (a, c)):
            key = (int(x), int(y)) if x < y else (int(y), int(x))
            edge_faces.setdefault(key, []).append(f)

    cos_threshold = np.cos(np.radians(crease_angle_deg))
    uf = _UnionFind(face_count)
    for face_list in edge_faces.values():
        if len(face_list) < 2: continue
        base = face_list[0]
        for other in face_list[1:]:
            if float(np.dot(unit_face_normals[base], unit_face_normals[other])) > cos_threshold:
                uf.union(base, other)

    pos_faces, vertex_faces = {}, [[] for _ in range(len(vertices))]
    for f in range(face_count):
        for corner in faces[f]: vertex_faces[corner].append(f)
        for p in face_pos[f]: pos_faces.setdefault(int(p), []).append(f)

    normals = np.zeros_like(vertices, dtype="f8")
    for v in range(len(vertices)):
        incident = vertex_faces[v]
        if not incident: continue
        my_roots = {uf.find(f) for f in incident}
        relevant = [f for f in pos_faces[int(pos_ids[v])] if uf.find(f) in my_roots]
        total = raw_face_normals[relevant].sum(axis=0)
        length = np.linalg.norm(total)
        normals[v] = total / length if length > 1e-12 else unit_face_normals[incident[0]]
    return normals.astype("f4")

def _compute_uvs(mesh, vertex_count):
    uvs = np.zeros((vertex_count, 2), dtype="f4")
    raw_uvs = getattr(mesh.visual, "uv", None)
    if raw_uvs is not None and len(raw_uvs) == vertex_count:
        uvs = np.asarray(raw_uvs, dtype="f4").copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]
    return uvs

def _normalize_color(values):
    col = np.asarray(values[:3], dtype="f4")
    return col / (255.0 if np.max(col) > 1.0 else 1.0)

def _upload_texture(ctx, img):
    if img is None: return None
    if not isinstance(img, Image.Image): img = Image.fromarray(np.asarray(img))
    img = img.convert("RGB")
    tex = ctx.texture(img.size, 3, img.tobytes())
    tex.build_mipmaps()
    # This filter setting is immediately overwritten every frame by
    # pbr_shader.py's _bind_material_textures (which sets .filter on
    # every bind call) - it's set here too only so this file doesn't
    # visually contradict what's actually in effect at runtime. If you
    # change the filtering mode, change it in _bind_material_textures;
    # this line won't do anything on its own.
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = True
    return tex

def _read_gltf_uv1(path, node_order):
    """Reads the TEXCOORD_1 (lightmap UV) accessor from each node in
    node_order and concatenates them in that same order - trimesh only
    exposes TEXCOORD_0, so this goes straight to the raw glTF data via
    pygltflib instead.

    node_order MUST be the exact same node-name sequence
    _flatten_scene concatenated vertices in (see load_glb, which
    threads it through from there) - matching pygltflib's raw node/
    mesh data against trimesh's flattened vertex order isn't a
    relationship that could be trusted to line up by re-deriving it
    independently here, so this only ever consumes an order someone
    else already established, rather than guessing at one. Originally
    this only handled a single-mesh, single-primitive glb at all
    (bailing out on anything else) - confirmed as the reason a model
    like plane.glb (a real floor mesh plus 2 collision-only helper
    meshes _flatten_scene already excludes from rendering - see
    COLLISION_ONLY_PREFIX) silently lost its lightmap: the file has 3
    primitives in total, so the old blanket "exactly 1 primitive or
    bail" check rejected it outright even though the ONE node that
    actually gets rendered has perfectly good TEXCOORD_1 data. Per-node
    lookup by name sidesteps that - a node not being rendered was never
    actually a reason to doubt the rendered one's own UVs.

    Each node must still resolve to a single-primitive mesh with a
    plain non-sparse VEC2 float TEXCOORD_1 accessor (what Blender's
    glTF exporter produces for a second UV map) - multiple primitives
    under one node still isn't handled, same "don't guess" reasoning as
    before, just correctly scoped to nodes that matter now. Also still
    requires the lightmap UV to actually be the *second* UV map in
    Blender (TEXCOORD_1 is whichever UV map is second in the mesh's UV
    map list at export time, not anything named-based). Returns None -
    caller falls back to "no lightmap UV" - the moment ANY node in
    node_order fails this, since a partial lightmap UV set covering
    only some of the rendered geometry would misalign texture coords
    across the rest."""
    try:
        from pygltflib import GLTF2
    except ImportError:
        print(
            "[model_loader] pygltflib is not installed - lightmap UVs "
            "can never be detected regardless of what's in the glb. "
            "Run: pip install pygltflib"
        )
        return None

    try:
        gltf = GLTF2().load(str(path))
        nodes_by_name = {n.name: n for n in (gltf.nodes or []) if n.name}
        blob = gltf.binary_blob()

        all_uv1 = []
        for name in node_order:
            node = nodes_by_name.get(name)
            if node is None or node.mesh is None: return None

            mesh = gltf.meshes[node.mesh]
            if len(mesh.primitives) != 1: return None

            idx = getattr(mesh.primitives[0].attributes, "TEXCOORD_1", None)
            if idx is None: return None

            acc = gltf.accessors[idx]
            if acc.sparse or acc.componentType != 5126 or acc.type != "VEC2": return None

            view = gltf.bufferViews[acc.bufferView]
            offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
            all_uv1.append(np.frombuffer(blob, dtype="<f4", count=acc.count * 2, offset=offset).reshape(-1, 2).copy())

        return np.concatenate(all_uv1, axis=0) if all_uv1 else None
    except Exception as e:
        print(f"[model_loader] Failed to read lightmap UV from {path}: {e}")
        return None

def _read_raw_gltf_material_factors(path):
    try:
        from pygltflib import GLTF2
        gltf = GLTF2().load(str(path))
        return {m.name: {"roughnessFactor": getattr(m.pbrMetallicRoughness, "roughnessFactor", None) if m.pbrMetallicRoughness else None,
                         "metallicFactor": getattr(m.pbrMetallicRoughness, "metallicFactor", None) if m.pbrMetallicRoughness else None}
                for m in (gltf.materials or [])}
    except Exception:
        return {}

def _extract_material(mesh, scene, ctx, raw_factors=None):
    base_color, metallic, roughness, emissive = np.array([0.8, 0.8, 0.8], dtype="f4"), 1.0, 1.0, np.zeros(3, dtype="f4")
    mat = getattr(mesh.visual, "material", None)
    if mat is None: return base_color, metallic, roughness, emissive, None, None

    for attr in ("main_color", "baseColorFactor", "diffuse"):
        val = getattr(mat, attr, None)
        if val is not None:
            base_color = _normalize_color(val)
            break

    file_factors = (raw_factors or {}).get(getattr(mat, "name", None), {})
    for k, target in [("metallicFactor", "metallic"), ("roughnessFactor", "roughness")]:
        val = getattr(mat, k, None) if getattr(mat, k, None) is not None else file_factors.get(k)
        if val is not None:
            try:
                if target == "metallic": metallic = float(val)
                else: roughness = float(val)
            except (TypeError, ValueError): pass

    em_val = getattr(mat, "emissiveFactor", None)
    if em_val is not None: emissive = _normalize_color(em_val)

    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    if img is None and isinstance(scene, trimesh.Scene):
        textures = getattr(scene, "textures", None)
        if textures:
            img = next(iter(textures.values()))

    return base_color, metallic, roughness, emissive, _upload_texture(ctx, img), _upload_texture(ctx, getattr(mat, "metallicRoughnessTexture", None))

def _extract_vertex_colors(mesh, base_color, vertex_count):
    colors = np.tile(base_color, (vertex_count, 1)).astype("f4")
    v_cols = getattr(mesh.visual, "vertex_colors", None)
    if v_cols is not None:
        v_cols = np.asarray(v_cols[:, :3], dtype="f4") / 255.0
        if len(v_cols) == vertex_count and not np.allclose(v_cols, v_cols[0]): colors = v_cols
    return colors

def _get_vertex_normals(mesh, vertices, faces, recompute_normals, crease_angle_deg):
    if not recompute_normals:
        file_normals = getattr(mesh, "vertex_normals", None)
        if file_normals is not None:
            file_normals = np.asarray(file_normals, dtype="f4")
            if file_normals.shape == vertices.shape:
                lengths = np.linalg.norm(file_normals, axis=1)
                if np.isfinite(file_normals).all() and np.all(lengths > 1e-6):
                    return file_normals / lengths[:, None]
    return _compute_vertex_normals(vertices, faces, crease_angle_deg)

def _flatten_scene(scene):
    """Returns (mesh_or_scene, node_order): node_order is the exact
    sequence of node names actually concatenated into the combined mesh
    (collision-only nodes excluded), so callers needing to re-derive
    per-vertex data from the raw glTF (e.g. lightmap TEXCOORD_1, which
    trimesh itself doesn't expose) can look nodes up by name in that
    same order. For a non-Scene input there's no per-node breakdown, so
    node_order comes back empty."""
    if not isinstance(scene, trimesh.Scene): return scene, []
    all_vertices, all_faces, all_normals, all_uvs, representative, vertex_offset = [], [], [], [], None, 0
    used_node_names = []
    for node_name in scene.graph.nodes_geometry:
        if _is_collision_only_node(node_name): continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if geom is None or not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0: continue
        rot, trans = transform[:3, :3], transform[:3, 3]
        all_vertices.append(np.asarray(geom.vertices) @ rot.T + trans)
        all_faces.append(np.asarray(geom.faces) + vertex_offset)

        geom_normals = getattr(geom, "vertex_normals", None)
        all_normals.append((np.asarray(geom_normals) @ rot.T) if geom_normals is not None and len(geom_normals) == len(geom.vertices) else None)

        geom_uv = getattr(geom.visual, "uv", None)
        all_uvs.append(np.asarray(geom_uv) if geom_uv is not None and len(geom_uv) == len(geom.vertices) else None)

        if representative is None: representative = geom
        vertex_offset += len(geom.vertices)
        used_node_names.append(node_name)

    if not all_vertices: return None, []
    combined = trimesh.Trimesh(vertices=np.concatenate(all_vertices), faces=np.concatenate(all_faces), process=False)
    if all(n is not None for n in all_normals): combined.vertex_normals = np.concatenate(all_normals, axis=0)
    if representative is not None:
        combined.visual = representative.visual
        if all(u is not None for u in all_uvs): combined.visual.uv = np.concatenate(all_uvs, axis=0)
    return combined, used_node_names

def load_glb(filepath, ctx, prog, recompute_normals=False, crease_angle_deg=DEFAULT_CREASE_ANGLE_DEG):
    path = Path(filepath)
    if not path.exists(): return print(f"[Warning] Model file not found: {path.resolve()}") or None
    buffers, vao, tex_obj, mr_tex_obj = [], None, None, None
    try:
        scene = trimesh.load(str(path), process=False)
        mesh, node_order = _flatten_scene(scene)
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0: raise RuntimeError("Invalid or empty mesh.")

        vertices, faces = np.asarray(mesh.vertices, dtype="f4"), np.asarray(mesh.faces, dtype="i4")
        normals = _get_vertex_normals(mesh, vertices, faces, recompute_normals, crease_angle_deg)
        uvs = _compute_uvs(mesh, len(vertices))

        lightmap_uvs = _read_gltf_uv1(path, node_order) if node_order else None
        has_lightmap_uv = lightmap_uvs is not None and len(lightmap_uvs) == len(vertices)
        if not has_lightmap_uv: lightmap_uvs = np.zeros((len(vertices), 2), dtype="f4")

        base_color, metallic, roughness, emissive, tex_obj, mr_tex_obj = _extract_material(mesh, scene, ctx, _read_raw_gltf_material_factors(path))
        colors = _extract_vertex_colors(mesh, base_color, len(vertices))

        attr_data = {"in_position": (vertices, "3f"), "in_normal": (normals, "3f"), "in_color": (colors, "3f"), "in_uv": (uvs, "2f"), "in_lightmap_uv": (lightmap_uvs, "2f")}
        vbos = {name: ctx.buffer(data.tobytes()) for name, (data, fmt) in attr_data.items() if _has_attribute(prog, name)}
        ibo = ctx.buffer(faces.tobytes())
        buffers = list(vbos.values()) + [ibo]

        vao_content = [(vbos[name], attr_data[name][1], name) for name in vbos]
        if not vao_content: raise RuntimeError("No recognized vertex attributes.")
        vao = ctx.vertex_array(prog, vao_content, ibo)

        return {
            "vao": vao, "vbo": vbos.get("in_position"), "normal_vbo": vbos.get("in_normal"),
            "color_vbo": vbos.get("in_color"), "uv_vbo": vbos.get("in_uv"),
            "lightmap_uv_vbo": vbos.get("in_lightmap_uv"), "has_lightmap_uv": has_lightmap_uv, "ibo": ibo,
            "texture": tex_obj, "metallic_roughness_texture": mr_tex_obj,
            "metallic": metallic, "roughness": roughness, "emissive": emissive.tolist(),
            "has_texture": 1 if tex_obj else 0, "has_metallic_roughness_texture": 1 if mr_tex_obj else 0,
        }
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        for res in [vao, *buffers, tex_obj, mr_tex_obj]:
            if res: res.release()
        return None