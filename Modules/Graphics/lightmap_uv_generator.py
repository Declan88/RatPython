"""
Procedural lightmap UV generation - a fallback for meshes that don't
ship an authored TEXCOORD_1 (lightmap UV) channel, similar in spirit to
Blender's own "Lightmap Pack" UV operator: group the mesh into
per-planar-island charts (not literally per-triangle - see
_build_charts for why that matters at real mesh triangle counts),
flatten each chart into 2D via a simple planar projection (exact/
distortion-free, since every triangle in a chart is coplanar-ish by
construction), then pack every chart's rectangle into the unit UV
square with a texel-padded margin so bilinear sampling during baking
never bleeds one chart's lighting into its neighbor's.

This is NOT a general-purpose unwrapper - it exists purely to make
lightmap_baker.py's bake_static_lighting() usable on assets that were
authored/exported without a lightmap UV set (see model_loader.py's
_read_gltf_uv1/_build_mesh_data - generate_lightmap_uvs() below is the
fallback called when that comes back empty). It deliberately does not
attempt a true conformal unwrap (LSCM/ABF-style parametrization with
seam-aware chart growing) - that's a much bigger undertaking than a
lightmap actually needs, which only cares about getting non-overlapping,
reasonably-packed islands with enough resolution to bake soft, blurry
static lighting into, not about preserving texture-space angles/areas
on a genuinely curved surface the way a real unwrap for a hand-painted
texture would need to.

Kept deliberately independent of the rest of this project (plain numpy
in, plain numpy out, no moderngl/trimesh/Scene imports) so it can be
tested and reasoned about on its own - see model_loader.py for the one
call site that actually wires it into the asset-loading pipeline.
"""

import numpy as np


class _UnionFind:
    """Path-compressed union-find over triangle indices - see
    _build_charts, which is the only user of this."""

    __slots__ = ("parent",)

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _face_normals_and_areas(vertices, faces):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    raw = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(raw, axis=1)
    areas = lengths * 0.5
    safe_lengths = np.where(lengths > 1e-12, lengths, 1.0)
    return raw / safe_lengths[:, None], areas


def _build_charts(vertices, faces, angle_threshold_deg):
    """Groups triangles into charts of mutually coplanar-ish, edge-
    connected triangles: union-find over the face-adjacency graph
    (shared-edge pairs), joining two triangles whenever their face
    normals are within angle_threshold_deg of each other.

    This is what keeps a flat wall/floor authored as many small
    triangles from exploding into one UV island per triangle the way a
    naive "one square per face" packer would (like model_loader.py's
    own _canonical_position_ids/_UnionFind crease-angle grouping for
    vertex normals, this is the same "merge across near-coplanar
    adjacent faces" idea, just applied to UV charting instead of
    normal smoothing) - while still splitting at genuine creases/
    corners: an adjacent pair whose normals disagree by more than the
    threshold never merges, so e.g. a box's 6 faces stay 6 separate
    charts even though every pair of adjacent faces shares an edge.

    Returns a list of int arrays, each the triangle indices belonging
    to one chart."""
    face_normals, _ = _face_normals_and_areas(vertices, faces)
    cos_threshold = np.cos(np.radians(angle_threshold_deg))

    edge_to_faces = {}
    for face_idx, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(u), int(v)) if u < v else (int(v), int(u))
            edge_to_faces.setdefault(key, []).append(face_idx)

    uf = _UnionFind(len(faces))
    for shared in edge_to_faces.values():
        if len(shared) != 2:
            # A boundary edge (only one triangle uses it) or a non-
            # manifold edge (more than two) - nothing well-defined to
            # merge across either way, leave both sides as-is.
            continue
        i, j = shared
        if np.dot(face_normals[i], face_normals[j]) >= cos_threshold:
            uf.union(i, j)

    charts = {}
    for face_idx in range(len(faces)):
        root = uf.find(face_idx)
        charts.setdefault(root, []).append(face_idx)
    return [np.array(tris, dtype="i8") for tris in charts.values()]


def _orthonormal_basis(normal):
    """Any two unit vectors perpendicular to `normal` and to each
    other - doesn't need to match any authored tangent, only to be
    consistent for every vertex flattened into the same chart."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, helper)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    return tangent, bitangent


def _flatten_chart(vertices, faces, chart_tris):
    """Projects one chart's triangles into a local, distortion-free 2D
    space (valid because every triangle in a chart is coplanar-ish by
    construction - see _build_charts) and re-indexes them onto their
    own compact vertex list: only the vertices this chart actually
    uses, deduplicated WITHIN the chart - never shared with any other
    chart's output vertices, since a mesh vertex sitting on a seam
    between two charts needs a different lightmap UV on each side.

    Returns (local_uvs (Nx2 float64), local_faces (Mx3 int, indexing
    local_uvs), original_vertex_indices (N,) int - original_vertex_
    indices[i] is which mesh vertex local_uvs[i] (and hence every new
    vertex i) was flattened from, for the caller to rebuild any other
    per-vertex array - positions, normals, uv0, vertex colors - to
    match by indexing with it)."""
    chart_faces = faces[chart_tris]
    unique_original, remapped_faces = np.unique(chart_faces, return_inverse=True)
    remapped_faces = remapped_faces.reshape(chart_faces.shape)

    positions = vertices[unique_original]
    face_normals, areas = _face_normals_and_areas(vertices, chart_faces)
    chart_normal = (face_normals * areas[:, None]).sum(axis=0)
    norm_len = np.linalg.norm(chart_normal)
    if norm_len < 1e-12:
        # Every triangle in this chart has ~zero area (a degenerate
        # sliver) - fall back to the first triangle's own normal
        # rather than dividing by zero building the weighted average.
        chart_normal = face_normals[0]
    else:
        chart_normal = chart_normal / norm_len

    tangent, bitangent = _orthonormal_basis(chart_normal)
    origin = positions[0]
    local = positions - origin
    local_uvs = np.stack([local @ tangent, local @ bitangent], axis=1)

    return local_uvs, remapped_faces, unique_original.astype("i8")


def _pack_charts(chart_uvs, margin):
    """Simple shelf/skyline packer: sorts charts tallest-first and lays
    them out left-to-right, wrapping to a new row whenever the next
    chart would exceed a target row width. Not space-optimal (a real
    bin-packer would do better), but simple, fast even for thousands of
    charts, and always produces a valid non-overlapping layout - which
    is all a lightmap atlas actually needs; wasted texel space just
    means a slightly coarser effective bake resolution, not a visible
    bug.

    margin: in the same local/world units as chart_uvs - the gap
    enforced around every chart's own edge (so between any two charts
    it doubles up to 2*margin), sized so a padded pixel in the
    eventual baked texture never samples a neighboring chart's bake
    (see generate_lightmap_uvs' own padding_texels docstring).

    Returns (offsets, (packed_width, packed_height)): offsets[i] is
    the (x, y) to add to chart_uvs[i] to place it in the shared atlas;
    the caller normalizes every placed coordinate by the packed extent
    to land in 0..1."""
    sizes = []
    for uv in chart_uvs:
        mins, maxs = uv.min(axis=0), uv.max(axis=0)
        sizes.append((maxs[0] - mins[0] + margin * 2.0, maxs[1] - mins[1] + margin * 2.0, mins))
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][1], reverse=True)

    # A roughly-square target row width keeps the packed layout from
    # becoming one absurdly long single row (or from being entirely
    # dominated by whichever single chart is biggest) - approximated
    # from total chart area, the same rough heuristic real texture
    # atlas packers use to pick a starting canvas size.
    total_area = sum(w * h for w, h, _ in sizes)
    target_row_width = max(float(np.sqrt(total_area)), max((w for w, _, _ in sizes), default=1.0))

    offsets = [None] * len(chart_uvs)
    cursor_x, cursor_y, shelf_height = 0.0, 0.0, 0.0
    packed_width, packed_height = 0.0, 0.0
    for i in order:
        w, h, mins = sizes[i]
        if cursor_x > 0.0 and cursor_x + w > target_row_width:
            cursor_x = 0.0
            cursor_y += shelf_height
            shelf_height = 0.0
        offsets[i] = (cursor_x + margin - mins[0], cursor_y + margin - mins[1])
        cursor_x += w
        shelf_height = max(shelf_height, h)
        packed_width = max(packed_width, cursor_x)
        packed_height = max(packed_height, cursor_y + shelf_height)

    return offsets, (max(packed_width, 1e-9), max(packed_height, 1e-9))


def generate_lightmap_uvs(vertices, faces, resolution=256, padding_texels=4.0,
                           angle_threshold_deg=5.0):
    """Generates a lightmap UV set for a mesh that doesn't have one -
    see this module's own docstring for the overall approach (planar-
    island charting + shelf packing, not a true general unwrap).

    vertices: (N, 3) float array - world or local space, either is
    fine, only relative positions WITHIN a chart matter (see
    _flatten_chart).
    faces: (M, 3) int array of triangle vertex indices into vertices.
    resolution: the lightmap texture resolution this UV set is
    expected to eventually be baked at (e.g. Scene.bake_static_
    lighting's own lightmap_resolution) - used only to convert
    padding_texels into a UV-space margin, never to allocate a texture
    here. The actual bake resolution is decided later, independently,
    at bake time (see Scene.add_static/bake_static_lighting) - if the
    real resolution ends up HIGHER than this, the margin is simply
    more generous than the minimum needed (safe); a resolution LOWER
    than this would under-pad and risk bleed, which is why the default
    here matches bake_static_lighting's own conservative default (256)
    rather than this project's typically-much-higher per-scene
    settings (e.g. MainMapScene's 4096) - pass the real target
    resolution through if it's known to be lower than 256.
    padding_texels: gap enforced between charts, in texels of the
    eventual bake target - needs to be at least as big as whatever
    blur/dilation radius lightmap_baker.py applies, or adjacent charts'
    baked lighting will visibly bleed into each other.
    angle_threshold_deg: see _build_charts - how much two adjacent
    triangles' normals may differ and still be merged into the same
    chart. Smaller = more, smaller, flatter charts (less distortion,
    worse packing efficiency); larger = fewer, bigger charts that
    increasingly approximate curved surfaces as flat (some real
    stretch distortion creeps in past a few degrees).

    Returns (new_faces, uv1, vertex_remap):
      new_faces: (M, 3) int32 array, same triangle count/order as the
        input `faces`, reindexed into the NEW (expanded) vertex list
        this function produces - charts never share vertices with each
        other (see _flatten_chart), so this is generally larger than
        `vertices`.
      uv1: (K, 2) float32 array clamped to 0..1, K == len(vertex_remap)
        == the new vertex count - the lightmap UV for each new vertex.
      vertex_remap: (K,) int32 array - vertex_remap[i] is which
        ORIGINAL mesh vertex new vertex i was duplicated/kept from.
        Rebuild any other per-vertex array (positions, normals, uv0,
        vertex colors, ...) to match the new vertex count via
        `original_array[vertex_remap]` - they must all be reindexed
        together with `new_faces`, a mesh can't mix an old-indexed
        attribute with new_faces' new indices.
    """
    vertices = np.asarray(vertices, dtype="f8")
    faces = np.asarray(faces, dtype="i8")
    if len(faces) == 0:
        return (
            np.zeros((0, 3), dtype="i4"),
            np.zeros((0, 2), dtype="f4"),
            np.zeros((0,), dtype="i4"),
        )

    charts = _build_charts(vertices, faces, angle_threshold_deg)

    chart_uvs, chart_faces_local, chart_vertex_sources = [], [], []
    for chart_tris in charts:
        local_uvs, local_faces, original_indices = _flatten_chart(vertices, faces, chart_tris)
        chart_uvs.append(local_uvs)
        chart_faces_local.append(local_faces)
        chart_vertex_sources.append(original_indices)

    margin = float(padding_texels) / float(max(resolution, 1))
    offsets, (packed_w, packed_h) = _pack_charts(chart_uvs, margin)
    atlas_extent = max(packed_w, packed_h)

    all_uv1, all_sources, all_faces = [], [], []
    vertex_cursor = 0
    for local_uvs, local_faces, original_indices, (ox, oy) in zip(
        chart_uvs, chart_faces_local, chart_vertex_sources, offsets
    ):
        placed = (local_uvs + np.array([ox, oy])) / atlas_extent
        all_uv1.append(placed)
        all_sources.append(original_indices)
        all_faces.append(local_faces + vertex_cursor)
        vertex_cursor += len(original_indices)

    uv1 = np.clip(np.concatenate(all_uv1, axis=0), 0.0, 1.0).astype("f4")
    vertex_remap = np.concatenate(all_sources, axis=0).astype("i4")
    new_faces = np.concatenate(all_faces, axis=0).astype("i4")

    return new_faces, uv1, vertex_remap
