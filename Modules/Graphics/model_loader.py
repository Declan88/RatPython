from pathlib import Path
import moderngl
import numpy as np
from PIL import Image
import trimesh
import glm

DEFAULT_CREASE_ANGLE_DEG = 30.0

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

def _as_pil_image(img):
    if img is None: return None
    if not isinstance(img, Image.Image): img = Image.fromarray(np.asarray(img))
    return img.convert("RGB")

def _upload_texture(ctx, img):
    img = _as_pil_image(img)
    if img is None: return None
    tex = ctx.texture(img.size, 3, img.tobytes())
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = tex.repeat_y = True
    return tex

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
    if not isinstance(scene, trimesh.Scene): return scene
    all_vertices, all_faces, all_normals, all_uvs, representative, vertex_offset = [], [], [], [], None, 0
    for node_name in scene.graph.nodes_geometry:
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

    if not all_vertices: return None
    combined = trimesh.Trimesh(vertices=np.concatenate(all_vertices), faces=np.concatenate(all_faces), process=False)
    if all(n is not None for n in all_normals): combined.vertex_normals = np.concatenate(all_normals, axis=0)
    if representative is not None:
        combined.visual = representative.visual
        if all(u is not None for u in all_uvs): combined.visual.uv = np.concatenate(all_uvs, axis=0)
    return combined

def load_glb(filepath, ctx, prog, recompute_normals=False, crease_angle_deg=DEFAULT_CREASE_ANGLE_DEG):
    path = Path(filepath)
    if not path.exists(): return print(f"[Warning] Model file not found: {path.resolve()}") or None
    buffers, vao, tex_obj, mr_tex_obj = [], None, None, None
    try:
        scene = trimesh.load(str(path), process=False)
        mesh = _flatten_scene(scene)
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0: raise RuntimeError("Invalid or empty mesh.")

        vertices, faces = np.asarray(mesh.vertices, dtype="f4"), np.asarray(mesh.faces, dtype="i4")
        normals = _get_vertex_normals(mesh, vertices, faces, recompute_normals, crease_angle_deg)
        uvs = _compute_uvs(mesh, len(vertices))
        base_color, metallic, roughness, emissive, tex_obj, mr_tex_obj = _extract_material(mesh, scene, ctx, _read_raw_gltf_material_factors(path))
        colors = _extract_vertex_colors(mesh, base_color, len(vertices))

        attr_data = {"in_position": (vertices, "3f"), "in_normal": (normals, "3f"), "in_color": (colors, "3f"), "in_uv": (uvs, "2f")}
        vbos = {name: ctx.buffer(data.tobytes()) for name, (data, fmt) in attr_data.items() if _has_attribute(prog, name)}
        ibo = ctx.buffer(faces.tobytes())
        buffers = list(vbos.values()) + [ibo]

        vao_content = [(vbos[name], attr_data[name][1], name) for name in vbos]
        if not vao_content: raise RuntimeError("No recognized vertex attributes.")
        vao = ctx.vertex_array(prog, vao_content, ibo)

        return {
            "vao": vao, "vbo": vbos.get("in_position"), "normal_vbo": vbos.get("in_normal"),
            "color_vbo": vbos.get("in_color"), "uv_vbo": vbos.get("in_uv"), "ibo": ibo,
            "texture": tex_obj, "metallic_roughness_texture": mr_tex_obj,
            "metallic": metallic, "roughness": roughness, "emissive": emissive.tolist(),
            "has_texture": 1 if tex_obj else 0, "has_metallic_roughness_texture": 1 if mr_tex_obj else 0,
        }
    except Exception as e:
        print(f"[Error] Failed to parse model {path}: {e}")
        for res in [vao, *buffers, tex_obj, mr_tex_obj]:
            if res: res.release()
        return None

_SHADOW_VERTEX_SHADER = "#version 330\nuniform mat4 u_light_mvp;\nin vec3 in_position;\nvoid main() { gl_Position = u_light_mvp * vec4(in_position, 1.0); }"
_SHADOW_FRAGMENT_SHADER = "#version 330\nvoid main() {}"

class CascadedShadowMap:
    def __init__(self, ctx, resolution=2048, cascade_count=3):
        self.ctx, self.resolution, self.cascade_count, self.num_cascades = ctx, resolution, cascade_count, cascade_count
        self.near, self.far = 0.1, 100.0
        self.depth_textures = [ctx.depth_texture((resolution, resolution)) for _ in range(cascade_count)]
        for tex in self.depth_textures:
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            tex.repeat_x = tex.repeat_y = False
        self.fbos = [ctx.framebuffer(depth_attachment=tex) for tex in self.depth_textures]
        self.light_mvps = [glm.mat4(1.0) for _ in range(cascade_count)]
        self.splits = [0.0 for _ in range(max(0, cascade_count - 1))]
        self.program = ctx.program(vertex_shader=_SHADOW_VERTEX_SHADER, fragment_shader=_SHADOW_FRAGMENT_SHADER)

    def _get_camera_basis(self, camera):
        inv = glm.inverse(camera.get_view_matrix())
        return glm.vec3(inv[3]), glm.normalize(glm.vec3(inv[0])), glm.normalize(glm.vec3(inv[1])), glm.normalize(-glm.vec3(inv[2]))

    def _get_frustum_corners(self, camera, near_d, far_d):
        proj = camera.get_projection_matrix()
        pos, right, up, forward = self._get_camera_basis(camera)
        px, py = float(proj[0][0]), float(proj[1][1])
        tan_half = np.tan(0.5 * (2.0 * np.arctan(1.0 / py) if abs(px) >= 1e-6 else np.radians(60.0)))
        aspect = (py / px) if abs(px) >= 1e-6 else 1.6
        corners = []
        for dist in (near_d, far_d):
            h, w, center = dist * tan_half, dist * tan_half * aspect, pos + forward * dist
            corners += [center - right * w - up * h, center + right * w - up * h, center + right * w + up * h, center - right * w + up * h]
        return corners

    def _fit_light_frustum(self, corners, light, world_up, curr_split):
        center = sum(corners, glm.vec3(0.0)) / float(len(corners))
        # Eye offset and near/far padding used to be a flat 50.0 world
        # units regardless of scene scale. On a small-scale scene (this
        # project's rat-sized rooms), that fixed padding could dwarf the
        # actual cascade content -- e.g. a cascade whose real depth span
        # is only a few units getting stretched to 100+ units of near/far
        # range once padded. That wastes most of the depth buffer's
        # precision on empty space, and the shadow bias constants in the
        # shader (tuned for a much tighter range) end up translating into
        # a huge world-space offset -- which looks exactly like peter-
        # panning (the shadow detaching from its caster). Padding
        # proportional to the cascade's own bounding-box depth keeps the
        # depth range sane regardless of world scale.
        light_view_probe = glm.lookAt(center + light * max(curr_split * 2.0, 1.0), center, world_up)
        probe_min, probe_max = glm.vec3(float("inf")), glm.vec3(float("-inf"))
        for corner in corners:
            pt = glm.vec3(light_view_probe * glm.vec4(corner, 1.0))
            probe_min, probe_max = glm.min(probe_min, pt), glm.max(probe_max, pt)
        depth_extent = max(probe_max.z - probe_min.z, 1.0)  # avoid a zero-size box
        padding = max(0.5, depth_extent * 0.5)

        light_view = glm.lookAt(center + light * max(curr_split * 2.0, padding), center, world_up)
        min_xyz, max_xyz = glm.vec3(float("inf")), glm.vec3(float("-inf"))
        for corner in corners:
            pt = glm.vec3(light_view * glm.vec4(corner, 1.0))
            min_xyz, max_xyz = glm.min(min_xyz, pt), glm.max(max_xyz, pt)
        xy_pad = max(1.0, (max_xyz.x - min_xyz.x) * 0.02)
        return glm.ortho(min_xyz.x - xy_pad, max_xyz.x + xy_pad, min_xyz.y - xy_pad, max_xyz.y + xy_pad, max(0.01, -max_xyz.z - padding), -min_xyz.z + padding) * light_view

    def update(self, camera, light_dir):
        self.near, self.far = max(float(getattr(camera, "near", 0.1)), 0.01), max(float(getattr(camera, "far", 100.0)), self.near + 1.0)
        cascade_splits = [0.75 * (self.near * (self.far / self.near) ** ((i + 1) / float(self.cascade_count))) + 
                          0.25 * (self.near + (self.far - self.near) * ((i + 1) / float(self.cascade_count))) 
                          for i in range(self.cascade_count)]
        self.splits = cascade_splits[:-1]
        light = glm.normalize(glm.vec3(*light_dir) if isinstance(light_dir, np.ndarray) else glm.vec3(light_dir.x, light_dir.y, light_dir.z))
        if glm.length(light) <= 1e-6: light = glm.vec3(0.5, 1.0, 0.8)
        world_up = glm.vec3(0.0, 0.0, 1.0) if abs(glm.dot(light, glm.vec3(0, 1, 0))) > 0.95 else glm.vec3(0, 1, 0)

        prev = self.near
        for i, split in enumerate(cascade_splits):
            self.light_mvps[i] = self._fit_light_frustum(self._get_frustum_corners(camera, prev, split), light, world_up, split)
            prev = split

    def render(self, render_callback):
        for i, fbo in enumerate(self.fbos):
            fbo.use()
            fbo.clear(depth=1.0)
            self.program["u_light_mvp"].write(self.light_mvps[i])
            render_callback(self.program)

    def destroy(self):
        for res in self.fbos + self.depth_textures + [self.program]:
            try: res.release()
            except Exception: pass