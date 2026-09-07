"""
Lightmap disk cache I/O.

Uses OpenEXR when available - the actual industry-standard format for
multi-channel float HDR image data, with compression modes (ZIP here)
tuned for smooth continuous-tone content like baked lighting, typically
beating generic DEFLATE-on-raw-floats by a real margin. Falls back to
compressed .npz automatically if the OpenEXR package isn't installed,
so this isn't a hard dependency - just a size optimization when present.

NOTE: this module is written directly from the current (v3.3+) official
OpenEXR python module's documented API (openexr.com/en/latest/python.html
and the project's own README), not verified by actually running it - no
network access was available to pip install and test OpenEXR itself.
The plain .npz fallback path is the one to trust unconditionally; test
the OpenEXR path specifically before relying on it.

Metadata (which resolutions a cached lightmap was baked at, so a bake
call with different resolution settings can tell a cache is stale
rather than silently reusing it) is stored differently per format:
- .npz: as extra arrays alongside "data", using numpy's native support.
- .exr: EXR header attributes aren't used here (the simplified File API's
  support for arbitrary custom attributes isn't confirmed), so metadata
  is stored in a small sidecar .json file instead.
"""

import json

try:
    import OpenEXR
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print(
        "[lightmap_cache_io] OpenEXR not installed (pip install OpenEXR) - "
        "lightmap cache will use compressed .npz instead, which is larger "
        "but works the same otherwise. Not required."
    )

import numpy as np


def lightmap_cache_path(lightmap_dir, index):
    """The primary cache file path for a given lightmap index. Callers
    should treat this as the one path to check for existence/mtime
    purposes; the .exr format's sidecar .json is an implementation
    detail handled internally by save/load below."""
    ext = "exr" if HAS_OPENEXR else "npz"
    return lightmap_dir / f"lightmap_{index}.{ext}"


def save_lightmap_cache(path, array, lightmap_resolution, point_shadow_resolution):
    """array: (H, W, 3) float16 numpy array."""
    if HAS_OPENEXR:
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        channels = {"RGB": array.astype(np.float16)}
        with OpenEXR.File(header, channels) as outfile:
            outfile.write(str(path))
        path.with_suffix(".json").write_text(json.dumps({
            "lightmap_resolution": lightmap_resolution,
            "point_shadow_resolution": point_shadow_resolution,
        }))
    else:
        np.savez_compressed(
            path, data=array,
            lightmap_resolution=lightmap_resolution,
            point_shadow_resolution=point_shadow_resolution,
        )


def load_lightmap_cache(path, lightmap_resolution, point_shadow_resolution):
    """Returns the cached (H, W, 3) float16 array if it exists AND
    matches the requested resolutions, else None (missing or stale -
    either way, the caller should treat it as a cache miss and rebake)."""
    if HAS_OPENEXR:
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("lightmap_resolution") != lightmap_resolution or \
           meta.get("point_shadow_resolution") != point_shadow_resolution:
            return None
        with OpenEXR.File(str(path)) as infile:
            return infile.channels()["RGB"].pixels.astype(np.float16)
    else:
        if not path.exists():
            return None
        with np.load(path) as npz:
            if int(npz["lightmap_resolution"]) != lightmap_resolution or \
               int(npz["point_shadow_resolution"]) != point_shadow_resolution:
                return None
            return npz["data"]