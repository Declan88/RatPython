from pathlib import Path
from functools import lru_cache
import moderngl
import numpy as np
from PIL import Image
import trimesh

from Modules.Graphics.lightmap_uv_generator import generate_lightmap_uvs

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

def _compute_tangents(vertices, normals, uvs, faces):
    """Per-vertex tangent space for normal mapping - a (len(vertices), 4)
    float32 array, xyz = tangent, w = handedness (+-1, see below) -
    ready to upload as a vec4 in_tangent vertex attribute (pbr_shader.py
    reconstructs the bitangent in the fragment shader as cross(N, T) * w,
    the standard glTF convention, rather than uploading a full 9-float
    TBN or a separate bitangent attribute).

    Standard per-triangle accumulation (Lengyel's method): for each
    triangle, solve for the tangent/bitangent directions that would
    reproduce its own edge vectors from its own UV deltas, then sum that
    onto each of its 3 vertices (shared vertices end up averaged across
    every triangle touching them, same spirit as vertex normal
    smoothing) before normalizing. This SMOOTH per-vertex result (not a
    per-fragment screen-space-derivative reconstruction, which was tried
    first here and produces a piecewise-constant basis per triangle,
    visibly discontinuous at every triangle edge - confirmed as the
    actual cause of visible seam lines on a normal-mapped surface) is
    what pbr_shader.py's vertex shader interpolates smoothly across a
    triangle exactly the way vertex normals already are.

    normals here must already be this mesh's final PER-VERTEX normals
    (post-smoothing/recompute - see _get_vertex_normals) - each
    accumulated tangent is Gram-Schmidt orthogonalized against ITS OWN
    vertex's normal, not re-derived from geometry, so tangent and normal
    are always perpendicular even where a smoothed normal doesn't
    exactly match any one triangle's own face normal."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    uv0, uv1, uv2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]

    edge1, edge2 = v1 - v0, v2 - v0
    duv1, duv2 = uv1 - uv0, uv2 - uv0

    denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    # A zero-area UV triangle (degenerate/duplicate UVs) has no defined
    # tangent direction - f=0 makes it contribute nothing to its 3
    # vertices' accumulation rather than dividing by zero into NaN/Inf.
    f = np.where(np.abs(denom) > 1e-12, 1.0 / np.where(denom == 0, 1.0, denom), 0.0)

    tri_tangent = f[:, None] * (duv2[:, 1:2] * edge1 - duv1[:, 1:2] * edge2)
    tri_bitangent = f[:, None] * (duv1[:, 0:1] * edge2 - duv2[:, 0:1] * edge1)

    tangent_accum = np.zeros_like(vertices)
    bitangent_accum = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(tangent_accum, faces[:, i], tri_tangent)
        np.add.at(bitangent_accum, faces[:, i], tri_bitangent)

    n_dot_t = np.sum(normals * tangent_accum, axis=1, keepdims=True)
    tangent = tangent_accum - normals * n_dot_t
    lengths = np.linalg.norm(tangent, axis=1, keepdims=True)

    # A vertex whose accumulated tangent came out (near-)zero (every
    # triangle touching it was degenerate above) falls back to an
    # arbitrary axis perpendicular to its normal, picking whichever of
    # two candidate world axes isn't nearly parallel to that normal so
    # the Gram-Schmidt below doesn't itself collapse to ~zero.
    use_alt = np.abs(normals[:, 0]) > 0.9
    fallback = np.where(
        use_alt[:, None],
        np.array([0.0, 0.0, 1.0], dtype="f4"),
        np.array([1.0, 0.0, 0.0], dtype="f4"),
    )
    fallback = fallback - normals * np.sum(normals * fallback, axis=1, keepdims=True)
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-8)

    safe = lengths[:, 0] > 1e-8
    tangent = np.where(safe[:, None], tangent / np.maximum(lengths, 1e-8), fallback)

    # Handedness: whether the ACTUAL accumulated bitangent direction
    # agrees with cross(N, T) or opposes it - needed so a mirrored UV
    # island (common in real authored assets - e.g. one UV-mirrored half
    # of a symmetric model) still perturbs the normal in the visually
    # correct direction rather than inverted on that half.
    handedness = np.where(
        np.sum(np.cross(normals, tangent) * bitangent_accum, axis=1) < 0.0, -1.0, 1.0
    ).astype("f4")

    return np.concatenate([tangent.astype("f4"), handedness[:, None]], axis=1)

def _normalize_color(values):
    col = np.asarray(values[:3], dtype="f4")
    return col / (255.0 if np.max(col) > 1.0 else 1.0)

def _upload_texture(ctx, img):
    if img is None: return None
    if not isinstance(img, Image.Image): img = Image.fromarray(np.asarray(img))
    # RGBA, not RGB - a base color texture's own alpha channel needs to
    # survive into the shader for MASK/BLEND alpha_mode materials to
    # render correctly (see _extract_material's own alpha_mode/
    # alpha_cutoff and pbr_shader.py's fragment shader). A texture with
    # no real alpha channel (a plain JPEG, say) just comes out fully
    # opaque (255) here, matching how it always rendered before this.
    img = img.convert("RGBA")
    tex = ctx.texture(img.size, 4, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = True
    return tex

@lru_cache(maxsize=8)
def _parse_gltf2_cached(path_str):
    """Parses a .glb's full glTF structure via pygltflib exactly once
    per file path for this process (an in-memory cache, not a disk
    one - nothing here is persisted, and nothing invalidates it if the
    file changes on disk mid-session, which never happens in this
    codebase's own load-once-at-scene-startup usage).

    _read_gltf_uv1 and _read_raw_gltf_material_factors both used to
    call GLTF2().load(path) independently, and _read_gltf_uv1 did so
    from INSIDE _build_mesh_data - i.e. once per MATERIAL GROUP, not
    once per file. pygltflib's own JSON->dataclass decoding (via
    dataclasses_json, which leans on Python's typing/reflection
    machinery per field) is slow enough on a large file that this
    was confirmed, via cProfile, to be a multi-second cost EACH call -
    for a 107-material glb, loaded 3x over (once per pbr/shadow/bake
    program - see _load_objects_by_material), that added up to several
    hundred redundant full-file re-parses and a multi-MINUTE scene
    load before anything appeared on screen. Raises ImportError if
    pygltflib isn't installed, same as the direct GLTF2().load(...)
    call this replaces - callers still handle that themselves for
    their own distinct warning message."""
    from pygltflib import GLTF2
    return GLTF2().load(path_str)


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
        gltf = _parse_gltf2_cached(str(path))
    except ImportError:
        print(
            "[model_loader] pygltflib is not installed - lightmap UVs "
            "can never be detected regardless of what's in the glb. "
            "Run: pip install pygltflib"
        )
        return None

    try:
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
        gltf = _parse_gltf2_cached(str(path))
        return {m.name: {"roughnessFactor": getattr(m.pbrMetallicRoughness, "roughnessFactor", None) if m.pbrMetallicRoughness else None,
                         "metallicFactor": getattr(m.pbrMetallicRoughness, "metallicFactor", None) if m.pbrMetallicRoughness else None,
                         # alphaMode/alphaCutoff aren't exposed by
                         # trimesh's own material wrapper at all (unlike
                         # roughness/metallic factors, which trimesh
                         # usually does surface) - straight from the raw
                         # glTF, same as these two. alphaMode's own glTF-
                         # spec default is "OPAQUE" when the field is
                         # omitted entirely; alphaCutoff's is 0.5,
                         # meaningful only for MASK.
                         "alphaMode": getattr(m, "alphaMode", None) or "OPAQUE",
                         "alphaCutoff": getattr(m, "alphaCutoff", None),
                         # Same story as alphaMode/alphaCutoff above -
                         # trimesh's own material wrapper doesn't surface
                         # this either, straight from the raw glTF. glTF
                         # spec default is False when omitted.
                         "doubleSided": bool(getattr(m, "doubleSided", False)),
                         # normalTexture.scale - trimesh's own material
                         # wrapper flattens normalTexture straight down
                         # to a bare PIL image (see _extract_material),
                         # dropping this factor entirely, so it's read
                         # from the raw glTF like the others above. glTF
                         # spec default is 1.0 (no extra intensity scale)
                         # when the texture reference is present but this
                         # field is omitted.
                         "normalScale": (
                             getattr(m.normalTexture, "scale", None) if m.normalTexture else None
                         )}
                for m in (gltf.materials or [])}
    except Exception:
        return {}

def _extract_material(mesh, scene, ctx, raw_factors=None, load_textures=True):
    """load_textures=False skips the two _upload_texture calls at the
    end entirely (returning (None, None) for tex_obj/mr_tex_obj instead)
    - every OTHER factor here (base_color/metallic/roughness/emissive/
    alpha_mode/alpha_cutoff) is cheap pure-Python/numpy work, but
    _upload_texture decodes a PIL image and uploads/mipmaps a GPU
    texture, measured at up to ~1s EACH on a large embedded image.
    _build_mesh_data passes False here for a `prog` that has no texture
    sampler at all (the depth-only shadow_program, and the lightmap
    bake_program - see that function's own comment) - their VAOs never
    read a texture, so decoding and uploading one for them was pure
    waste, confirmed as the actual cause of a ~2-6 MINUTE load on a
    107-material glb (each material's base-color/metallic-roughness
    textures were being decoded and uploaded to the GPU three times
    over - once per program - for only one of those three to ever
    sample them)."""
    base_color, metallic, roughness, emissive = np.array([0.8, 0.8, 0.8], dtype="f4"), 1.0, 1.0, np.zeros(3, dtype="f4")
    base_alpha, alpha_mode, alpha_cutoff, double_sided, normal_scale = 1.0, "OPAQUE", 0.5, False, 1.0
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return (base_color, metallic, roughness, emissive, base_alpha, alpha_mode, alpha_cutoff,
                double_sided, normal_scale, None, None, None)
    if not load_textures:
        return (base_color, metallic, roughness, emissive, base_alpha, alpha_mode, alpha_cutoff,
                double_sided, normal_scale, None, None, None)

    for attr in ("main_color", "baseColorFactor", "diffuse"):
        val = getattr(mat, attr, None)
        if val is not None:
            base_color = _normalize_color(val)
            # The 4th (alpha) component, if the source array actually
            # has one - _normalize_color above only ever looks at the
            # first 3. Values already in 0..1 (a raw glTF baseColorFactor)
            # pass through as-is; an 8-bit 0..255 channel (some trimesh
            # material representations) gets normalized the same way
            # _normalize_color does for RGB.
            if len(val) > 3:
                raw_a = float(val[3])
                base_alpha = raw_a / 255.0 if raw_a > 1.0 else raw_a
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

    # alphaMode/alphaCutoff only ever come from the raw glTF read (see
    # _read_raw_gltf_material_factors) - trimesh's own material wrapper
    # doesn't surface either.
    if file_factors.get("alphaMode") in ("OPAQUE", "MASK", "BLEND"):
        alpha_mode = file_factors["alphaMode"]
    if file_factors.get("alphaCutoff") is not None:
        try: alpha_cutoff = float(file_factors["alphaCutoff"])
        except (TypeError, ValueError): pass
    # doubleSided only ever comes from the raw glTF read too, same
    # reasoning as alphaMode/alphaCutoff above - Scene._render_scene
    # reads this back (see this function's own return value and _build_
    # mesh_data's "double_sided" key) to skip back-face culling for an
    # otherwise-OPAQUE material that explicitly opts into double-sided
    # rendering (e.g. a water plane meant to be seen from both above and
    # below), the same way MASK already unconditionally does.
    double_sided = bool(file_factors.get("doubleSided", False))
    if file_factors.get("normalScale") is not None:
        try: normal_scale = float(file_factors["normalScale"])
        except (TypeError, ValueError): pass

    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    if img is None and isinstance(scene, trimesh.Scene):
        textures = getattr(scene, "textures", None)
        if textures:
            img = next(iter(textures.values()))

    # normalTexture - trimesh DOES flatten this straight to a bare PIL
    # image (unlike alphaMode/alphaCutoff/doubleSided/normalScale above,
    # which it drops entirely) - see this function's own module-level
    # confirmation via a live trimesh load. _upload_texture is reused
    # as-is (not given any special linear/non-sRGB treatment) because
    # this pipeline never applies a GPU-side sRGB internal format to
    # BEGIN with - the fragment shader manually un-gammas the base color
    # sample instead (see pbr_shader.py's pow(raw_albedo, vec3(2.2))) -
    # so a plain RGBA upload is already correct for normal data too, as
    # long as the shader does NOT apply that same pow(2.2) to it (it
    # doesn't - see FRAGMENT_SHADER_BODY's normal-map sampling).
    normal_tex_obj = _upload_texture(ctx, getattr(mat, "normalTexture", None))

    return (
        base_color, metallic, roughness, emissive, base_alpha, alpha_mode, alpha_cutoff, double_sided,
        normal_scale, _upload_texture(ctx, img), _upload_texture(ctx, getattr(mat, "metallicRoughnessTexture", None)),
        normal_tex_obj,
    )

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

def _build_mesh_data(mesh, node_order, path, scene, ctx, prog, recompute_normals, crease_angle_deg, raw_factors, load_textures=None):
    """The GPU-upload half of load_glb - given an already-flattened
    (mesh, node_order) pair (see _flatten_scene/_flatten_scene_by_
    material) plus the file's raw material factors (see
    _read_raw_gltf_material_factors, computed once by the caller and
    passed in here rather than re-parsed per call - matters for load_glb_
    by_material, which calls this once per material group from the same
    file), returns the same dict shape load_glb always has, or None on
    failure (after releasing whatever GL resources this call already
    created, exactly as load_glb's own try/except used to do inline)."""
    buffers, vao, tex_obj, mr_tex_obj, normal_tex_obj = [], None, None, None, None
    try:
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0: raise RuntimeError("Invalid or empty mesh.")

        vertices, faces = np.asarray(mesh.vertices, dtype="f4"), np.asarray(mesh.faces, dtype="i4")
        normals = _get_vertex_normals(mesh, vertices, faces, recompute_normals, crease_angle_deg)
        uvs = _compute_uvs(mesh, len(vertices))

        lightmap_uvs = _read_gltf_uv1(path, node_order) if node_order else None
        has_lightmap_uv = lightmap_uvs is not None and len(lightmap_uvs) == len(vertices)

        # Only the actual PBR/color-pass program's own VAO build needs a
        # decoded/uploaded texture - the lightmap bake_program's shader
        # declares no u_texture at all, and shadow_program's now DOES
        # (see its own fragment shader in scene_base.py - an alpha-
        # tested discard for MASK/BLEND casters) but never uses one
        # loaded via THIS path: Scene._bind_shadow_alpha rebinds the
        # object's already-loaded PBR-pass texture at draw time instead
        # (see that method's own docstring), so a second, separate
        # decode/upload just for the shadow VAO build would be pure
        # duplicated waste - confirmed via cProfile as a real, multi-
        # second-per-material cost earlier in this project's history
        # (see _parse_gltf2_cached's own docstring) once a scene grew to
        # 100+ materials, which is exactly the class of regression
        # skipping this explicitly avoids reintroducing. load_textures=
        # None (every existing caller) keeps the automatic "does prog
        # declare u_texture" heuristic; a caller building specifically
        # the SHADOW VAO passes load_textures=False explicitly to opt
        # out of that heuristic.
        needs_textures = ("u_texture" in prog) if load_textures is None else load_textures
        (
            base_color, metallic, roughness, emissive, base_alpha, alpha_mode, alpha_cutoff, double_sided,
            normal_scale, tex_obj, mr_tex_obj, normal_tex_obj,
        ) = _extract_material(
            mesh, scene, ctx, raw_factors, load_textures=needs_textures
        )
        colors = _extract_vertex_colors(mesh, base_color, len(vertices))

        if not has_lightmap_uv:
            # No authored TEXCOORD_1 - generate one instead of falling
            # back to an all-zero UV (which used to just leave the
            # object out of lightmap baking entirely - see has_
            # lightmap_uv's other read sites in scene_base.py). See
            # lightmap_uv_generator.py's own docstring for the actual
            # unwrap approach. This RE-INDEXES the mesh (charts can't
            # share vertices across a UV seam - see generate_lightmap_
            # uvs' own vertex_remap docstring), so every other per-
            # vertex array has to be rebuilt to match via the same
            # vertex_remap before anything below reads len(vertices)
            # again.
            faces, lightmap_uvs, vertex_remap = generate_lightmap_uvs(vertices, faces)
            vertices = vertices[vertex_remap]
            normals = normals[vertex_remap]
            uvs = uvs[vertex_remap]
            colors = colors[vertex_remap]
            has_lightmap_uv = True

        attr_data = {"in_position": (vertices, "3f"), "in_normal": (normals, "3f"), "in_color": (colors, "3f"), "in_uv": (uvs, "2f"), "in_lightmap_uv": (lightmap_uvs, "2f")}
        # Only pbr_program's own vertex shader declares in_tangent at all
        # (shadow_program/bake_program use their own separate, unrelated
        # vertex shaders - see scene_base.py - so _has_attribute is
        # always False for them here) - same "don't build data a VAO
        # won't even read" reasoning as needs_textures above, and
        # _compute_tangents is real per-triangle numpy work, not free.
        # Computed AFTER the lightmap-UV remap above (not before) so it
        # always matches whatever the FINAL vertices/faces/uvs/normals
        # arrays actually are.
        if _has_attribute(prog, "in_tangent"):
            attr_data["in_tangent"] = (_compute_tangents(vertices, normals, uvs, faces), "4f")
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
            "tangent_vbo": vbos.get("in_tangent"),
            "texture": tex_obj, "metallic_roughness_texture": mr_tex_obj, "normal_texture": normal_tex_obj,
            "metallic": metallic, "roughness": roughness, "emissive": emissive.tolist(),
            "normal_scale": normal_scale,
            "has_texture": 1 if tex_obj else 0, "has_metallic_roughness_texture": 1 if mr_tex_obj else 0,
            "has_normal_texture": 1 if normal_tex_obj else 0,
            # "OPAQUE" | "MASK" | "BLEND" (glTF alphaMode - see
            # _extract_material's own docstring) plus the cutoff MASK
            # uses and the material's own base alpha factor (multiplied
            # with the base color texture's own alpha, if any, in the
            # fragment shader - see pbr_shader.py). Scene._render_scene
            # uses alpha_mode to decide per-object blend/cull-face state:
            # OPAQUE renders exactly as before (back-face culled, no
            # blending); MASK/BLEND both render double-sided (no
            # culling) per this project's own choice to treat "not fully
            # opaque" as "render both sides" - MASK additionally discards
            # below alpha_cutoff in the shader instead of blending; BLEND
            # draws in a separate, depth-write-disabled pass after
            # every opaque/cutout object, with real alpha blending. An
            # OPAQUE material can ALSO ask for double-sided rendering via
            # glTF's own separate doubleSided flag (e.g. a water plane
            # meant to be seen from both above and below) - see
            # "double_sided" below, read independently of alpha_mode.
            "alpha_mode": alpha_mode, "alpha_cutoff": alpha_cutoff, "base_alpha": base_alpha,
            "double_sided": double_sided,
        }
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        for res in [vao, *buffers, tex_obj, mr_tex_obj, normal_tex_obj]:
            if res: res.release()
        return None

def load_glb(filepath, ctx, prog, recompute_normals=False, crease_angle_deg=DEFAULT_CREASE_ANGLE_DEG,
              load_textures=None):
    path = Path(filepath)
    if not path.exists(): return print(f"[Warning] Model file not found: {path.resolve()}") or None
    try:
        scene = trimesh.load(str(path), process=False)
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        return None
    mesh, node_order = _flatten_scene(scene)
    return _build_mesh_data(
        mesh, node_order, path, scene, ctx, prog, recompute_normals, crease_angle_deg,
        _read_raw_gltf_material_factors(path), load_textures=load_textures,
    )

def _material_key(geom):
    """Identity key for grouping geometry by material in
    _flatten_scene_by_material - the material's own name (glTF materials
    loaded through trimesh carry the name authored in Blender/the source
    file) when it has one, else its Python identity so two distinct
    unnamed materials never accidentally merge. None (no material at
    all) is its own valid group too."""
    mat = getattr(geom.visual, "material", None)
    if mat is None: return None
    name = getattr(mat, "name", None)
    return name if name else id(mat)

def _flatten_scene_by_material(scene):
    """Like _flatten_scene, but groups nodes by MATERIAL instead of
    merging every node in the file into one mesh - _flatten_scene keeps
    only the FIRST node's material for the whole combined result
    (`combined.visual = representative.visual`), which is invisible for
    a single-material prop (there's only one material to keep anyway)
    but silently wrong for a real multi-material level: confirmed via
    mainmap.glb, which has 9 distinct materials across 11 mesh nodes -
    the old single-flatten path rendered the whole map with whichever
    ONE material happened to belong to the first node in the file,
    including cases where that material has no texture at all.

    Returns a list of (combined_mesh, node_order) pairs, one per
    distinct material actually present (collision-only nodes excluded,
    same as _flatten_scene), in first-encountered order - each pair is
    exactly what _flatten_scene would have returned if the file had
    ONLY that material's nodes in it. A non-Scene input (a bare Trimesh/
    PointCloud with no per-node material split possible) falls back to
    a single one-group result, matching _flatten_scene's own behavior
    for that case."""
    if not isinstance(scene, trimesh.Scene):
        return [(scene, [])] if scene is not None else []

    groups = {}
    order = []
    for node_name in scene.graph.nodes_geometry:
        if _is_collision_only_node(node_name): continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if geom is None or not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0: continue

        key = _material_key(geom)
        if key not in groups:
            groups[key] = {"vertices": [], "faces": [], "normals": [], "uvs": [], "representative": None, "offset": 0, "names": []}
            order.append(key)
        g = groups[key]

        rot, trans = transform[:3, :3], transform[:3, 3]
        g["vertices"].append(np.asarray(geom.vertices) @ rot.T + trans)
        g["faces"].append(np.asarray(geom.faces) + g["offset"])

        geom_normals = getattr(geom, "vertex_normals", None)
        g["normals"].append((np.asarray(geom_normals) @ rot.T) if geom_normals is not None and len(geom_normals) == len(geom.vertices) else None)

        geom_uv = getattr(geom.visual, "uv", None)
        g["uvs"].append(np.asarray(geom_uv) if geom_uv is not None and len(geom_uv) == len(geom.vertices) else None)

        if g["representative"] is None: g["representative"] = geom
        g["offset"] += len(geom.vertices)
        g["names"].append(node_name)

    results = []
    for key in order:
        g = groups[key]
        if not g["vertices"]: continue
        combined = trimesh.Trimesh(vertices=np.concatenate(g["vertices"]), faces=np.concatenate(g["faces"]), process=False)
        if all(n is not None for n in g["normals"]): combined.vertex_normals = np.concatenate(g["normals"], axis=0)
        if g["representative"] is not None:
            combined.visual = g["representative"].visual
            if all(u is not None for u in g["uvs"]): combined.visual.uv = np.concatenate(g["uvs"], axis=0)
        results.append((combined, g["names"]))
    return results

def load_mesh_groups_by_material(model_path, scale=None):
    """Geometry-only per-material split of model_path - no GPU context,
    no texture/material-factor loading, nothing but (material_name,
    vertices, faces) triples. Built for PhysicsWorld.add_static_mesh_
    by_material (see its own docstring): physics_world.py already loads
    a collision mesh independently through a second, cheap trimesh pass
    (its own _load_mesh) rather than reusing the GPU-focused render
    load, and reuses THIS module's own _flatten_scene_by_material to
    split it by material rather than re-deriving "which triangles
    belong to which material" a second, potentially-drifting way -
    exactly the same reasoning load_glb_by_material already follows for
    the render side.

    scale: optional array-like (anything np.asarray accepts - a glm.vec3
    works fine despite this module not importing glm itself, same as
    physics_world.py's own _load_mesh takes a plain array-like rather
    than requiring a specific vector type) - multiplies every vertex
    position, matching _load_mesh's own convention.

    Returns a list of (material_name, vertices, faces) tuples, one per
    distinct material actually present (collision-only nodes excluded,
    matching every other loader in this file) - material_name is
    whatever _material_key resolves to (the glTF material's own
    authored name, an opaque id() for an unnamed one, or None for
    geometry with no material at all). Empty list if model_path doesn't
    exist or fails to parse."""
    path = Path(model_path)
    if not path.exists():
        print(f"[Warning] Model file not found: {path.resolve()}")
        return []
    try:
        scene = trimesh.load(str(path), process=False)
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        return []

    scale_arr = np.asarray(scale, dtype="f8") if scale is not None else None
    results = []
    for mesh, _node_names in _flatten_scene_by_material(scene):
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        vertices = np.asarray(mesh.vertices, dtype="f8")
        if scale_arr is not None:
            vertices = vertices * scale_arr
        results.append((_material_key(mesh), vertices, np.asarray(mesh.faces, dtype="i4")))
    return results

def load_mesh_groups_by_object(model_path, scale=None):
    """Geometry-only per-OBJECT (glTF node) split of model_path - like
    load_mesh_groups_by_material, but keyed by the mesh's own NODE name
    (Blender's "Object name", carried through to the node name in the
    exported glTF) instead of its material. A level is very often built
    by reusing a handful of trim materials across many distinct objects
    (a shared "concrete" material used by a dozen different floor/wall
    pieces, say) - collision/gameplay decisions ("this ONE object has no
    collision", "this ONE object plays a different footstep sound") are
    naturally per-OBJECT in that case, not per-material, which is why
    this exists as a genuinely separate function rather than a mode
    flag on load_mesh_groups_by_material: the two group geometry along
    completely different axes, and a node's OWN material is irrelevant
    here (two objects sharing one material still get two independent
    entries; a single object made of several materials would need
    load_mesh_groups_by_material instead, or authoring it as separate
    objects to begin with).

    Never merges across nodes - a node here is already the smallest
    addressable unit this can name at all, so unlike load_mesh_groups_
    by_material's per-material merging, there's nothing TO merge.
    Collision-only nodes are excluded, matching every other loader in
    this file. scale: same convention as load_mesh_groups_by_material's
    own (any np.asarray-compatible array-like).

    Returns a list of (node_name, vertices, faces) tuples, one per mesh
    node actually present. A file with no per-node structure at all (a
    bare Trimesh/PointCloud, not a trimesh.Scene) falls back to a single
    (None, vertices, faces) entry, matching _flatten_scene_by_material's
    own fallback for that case. Empty list if model_path doesn't exist
    or fails to parse."""
    path = Path(model_path)
    if not path.exists():
        print(f"[Warning] Model file not found: {path.resolve()}")
        return []
    try:
        scene = trimesh.load(str(path), process=False)
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        return []

    scale_arr = np.asarray(scale, dtype="f8") if scale is not None else None

    if not isinstance(scene, trimesh.Scene):
        if scene is None or len(scene.vertices) == 0:
            return []
        vertices = np.asarray(scene.vertices, dtype="f8")
        if scale_arr is not None:
            vertices = vertices * scale_arr
        return [(None, vertices, np.asarray(scene.faces, dtype="i4"))]

    results = []
    for node_name in scene.graph.nodes_geometry:
        if _is_collision_only_node(node_name):
            continue
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if geom is None or not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
            continue

        rot, trans = transform[:3, :3], transform[:3, 3]
        vertices = np.asarray(geom.vertices, dtype="f8") @ rot.T + trans
        if scale_arr is not None:
            vertices = vertices * scale_arr
        results.append((node_name, vertices, np.asarray(geom.faces, dtype="i4")))
    return results

def load_glb_by_material(filepath, ctx, prog, recompute_normals=False, crease_angle_deg=DEFAULT_CREASE_ANGLE_DEG,
                          load_textures=None):
    """Like load_glb, but returns a LIST of per-material mesh-data dicts
    (each shaped exactly like load_glb's own single return value, empty
    list on total failure) instead of one dict merging every material in
    the file into one - see _flatten_scene_by_material's own docstring
    for why this exists. _read_raw_gltf_material_factors(path) is read
    once here and shared across every group's _build_mesh_data call,
    rather than re-parsing the file's raw glTF once per material just
    for that lookup."""
    path = Path(filepath)
    if not path.exists(): return print(f"[Warning] Model file not found: {path.resolve()}") or []
    try:
        scene = trimesh.load(str(path), process=False)
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        return []
    groups = _flatten_scene_by_material(scene)
    if not groups:
        print(f"[Error] Failed to parse model {path}: no renderable geometry.")
        return []
    raw_factors = _read_raw_gltf_material_factors(path)
    results = []
    for mesh, node_order in groups:
        data = _build_mesh_data(
            mesh, node_order, path, scene, ctx, prog, recompute_normals, crease_angle_deg, raw_factors,
            load_textures=load_textures,
        )
        if data is not None:
            # Same key _flatten_scene_by_material itself grouped this
            # mesh by (name if the material has one, else id(mat) - see
            # _material_key) - exposed here so a caller (Scene.add_
            # static's own alpha_mode_overrides) can target a SPECIFIC
            # material by name without needing to re-export the source
            # file just to fix how one material's alphaMode was
            # authored (e.g. a foliage material exported as BLEND/
            # Alpha Blend in Blender when it should behave like
            # Unreal's Masked - hard cutout, full depth write/test, no
            # sort-order dependence - see that override's own docstring).
            data["material_name"] = _material_key(mesh)
            results.append(data)
    return results