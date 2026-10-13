"""
Disk cache for generate_lightmap_uvs' own output (lightmap_uv_generator.py)
- the UV ATLAS/chart LAYOUT itself (which faces got re-indexed into which
chart, and where each chart landed in the packed 0-1 square), not the
baked LIGHTING result lightmap_cache_io.py caches (a separate, later
stage - that one depends on the scene's actual lights too, this one only
on the mesh's own geometry).

Exists because generate_lightmap_uvs' chart-building + packing pass is
real per-triangle work that otherwise reran from scratch on EVERY single
scene construction, for every mesh that ships no authored TEXCOORD_1 -
confirmed as the dominant remaining cost once recalculate_shadows=False
already skips the lighting bake itself (mainmap.glb: 88 meshes, 183k+
vertices, zero with an authored lightmap UV channel).

Keyed purely by (model_path, node_order, vertex_count, face_count) - NOT
by which Scene subclass loaded it, since the unwrap only depends on the
mesh's own geometry, not on anything scene-specific. One shared cache
directory, independent of lightmap_cache_io.py's own per-scene-class
self.lightmap_dir, so the same mainmap.glb load is cached once and
reused regardless of which Scene happens to load it.

vertex_count/face_count are the INPUT mesh's counts (before generate_
lightmap_uvs' own re-indexing, which changes both) - the cheap
staleness check: a re-export that changes the mesh's geometry almost
certainly changes at least one of these, which is enough to treat the
cache as stale and regenerate rather than serving a mismatched layout.
This is deliberately not a full content hash of the actual vertex
data - too slow to be worth it for what's just a defensive secondary
check alongside the already content-derived cache filename.
"""

import hashlib
from pathlib import Path

import numpy as np

_CACHE_DIR = Path("Assets/Lightmaps/_uv_cache")


def _cache_path(model_path, node_order, vertex_count, face_count):
    raw = f"{model_path}|{','.join(node_order)}|{vertex_count}|{face_count}".encode("utf-8")
    key = hashlib.sha1(raw).hexdigest()
    return _CACHE_DIR / f"{key}.npz"


def load_uv_cache(model_path, node_order, vertex_count, face_count):
    """Returns (faces, lightmap_uvs, vertex_remap) from a cached
    generate_lightmap_uvs call with these exact inputs, or None (missing
    or stale - either way, the caller should treat it as a cache miss and
    regenerate)."""
    path = _cache_path(model_path, node_order, vertex_count, face_count)
    if not path.exists():
        return None
    with np.load(path) as npz:
        if int(npz["vertex_count"]) != vertex_count or int(npz["face_count"]) != face_count:
            return None
        return npz["faces"], npz["lightmap_uvs"], npz["vertex_remap"]


def save_uv_cache(model_path, node_order, vertex_count, face_count, faces, lightmap_uvs, vertex_remap):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model_path, node_order, vertex_count, face_count)
    np.savez_compressed(
        path, faces=faces, lightmap_uvs=lightmap_uvs, vertex_remap=vertex_remap,
        vertex_count=vertex_count, face_count=face_count,
    )
