"""
Disk cache for a fully-BUILT Bullet collision shape (BulletTriangleMeshShape),
not just the raw (vertices, faces) that went into it - _build_triangle_
mesh_shape's own per-triangle Python addTriangle() loop (physics_world.py)
is the actual cost (confirmed via direct timing: ~13s for mainmap.glb's
138k faces, with the trimesh reload that produces vertices/faces itself
taking under 0.1s), so caching raw mesh data wouldn't help at all - this
caches the already-built shape OBJECT instead, via Panda3D's own generic
BAM (Binary Asset Model) serialization every TypedWritable, including a
Bullet shape, inherits (encodeToBamStream/decodeFromBamStream) - a direct
binary dump/restore of the finished C++ object, confirmed ~165x faster to
decode than to rebuild from scratch (13.1s build vs 0.08s decode, same
mainmap.glb mesh).

Same recalculate_shadows-driven policy as lightmap_cache_io.py's own
bake cache: True rebuilds (from the per-triangle loop) and overwrites the
cache; False tries the cache first and only falls back to rebuilding (and
then still saves, so a later run gets the speedup) if nothing valid is
there yet.

Keyed by (model_path, discriminator, scale, vertex_count, face_count) -
discriminator distinguishes multiple shapes built from the same file
(add_static_mesh_by_material/by_object each build one shape PER group,
not one for the whole file) - a material or object name, or None for
add_static_mesh's own single whole-file shape. vertex_count/face_count
are the same cheap staleness check lightmap_uv_cache_io.py already uses -
not a full content hash, just enough to catch a re-export that changed
the geometry.
"""

import hashlib
from pathlib import Path

_CACHE_DIR = Path("Assets/Physics/_collision_cache")


def _cache_path(model_path, discriminator, scale, vertex_count, face_count):
    raw = f"{model_path}|{discriminator}|{tuple(scale) if scale is not None else None}|{vertex_count}|{face_count}"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{key}.bam"


def load_shape_cache(shape_cls, model_path, discriminator, scale, vertex_count, face_count):
    """shape_cls: the Bullet shape class to decode as (e.g.
    BulletTriangleMeshShape) - decodeFromBamStream is a classmethod, so
    the caller's own expected type picks which one gets called. Returns
    the decoded shape, or None (missing cache - the caller should build
    and save one instead)."""
    path = _cache_path(model_path, discriminator, scale, vertex_count, face_count)
    if not path.exists():
        return None
    return shape_cls.decodeFromBamStream(path.read_bytes())


def save_shape_cache(shape, model_path, discriminator, scale, vertex_count, face_count):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model_path, discriminator, scale, vertex_count, face_count)
    path.write_bytes(shape.encodeToBamStream())
