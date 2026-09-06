"""
Converts a baked lightmap .npy (RGBA16F, linear HDR) into a viewable PNG,
using the same Reinhard tonemap + gamma the runtime shader applies, so you
can visually check what actually got baked - independent of anything the
runtime rendering path might be doing wrong.

Usage:
    python view_lightmap.py Assets/Lightmaps/TorusScene/lightmap_0.npy
"""

import sys
import numpy as np
from PIL import Image


def main():
    if len(sys.argv) != 2:
        print("Usage: python view_lightmap.py <path-to-lightmap.npy>")
        return

    path = sys.argv[1]
    array = np.load(path).astype(np.float32)  # (H, W, 4) - RGBA, alpha unused

    rgb = array[:, :, :3]

    print(f"Loaded {path}: shape={array.shape}, dtype was float16")
    print(f"Raw linear value range: min={rgb.min():.4f}, max={rgb.max():.4f}, mean={rgb.mean():.4f}")

    # Same tonemap as pbr_shader.py's main(): Reinhard, then gamma 1/2.2.
    tonemapped = rgb / (rgb + 1.0)
    tonemapped = np.power(np.clip(tonemapped, 0.0, 1.0), 1.0 / 2.2)

    out = (tonemapped * 255).astype(np.uint8)
    out_path = path.rsplit(".", 1)[0] + "_preview.png"
    Image.fromarray(out, "RGB").save(out_path)
    print(f"Saved viewable preview to {out_path}")


if __name__ == "__main__":
    main()