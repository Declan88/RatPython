"""
The player-chosen rat fur color. It's a plain (r, g, b) tuple of 0-1 floats
everywhere in-game (None = the rat's own colors) and travels over the network
as a 6-digit hex string ("" = none), applied through Scene.set_skeletal_tint
with the mask texture below.
"""

import colorsys

RAT_TINT_MASK_PATH = "Assets/Textures/Rat/rat_d_c.png"


def hsv_to_rgb(h, s, v):
    return colorsys.hsv_to_rgb(h % 1.0, min(1.0, max(0.0, s)), min(1.0, max(0.0, v)))


def rgb_to_hsv(rgb):
    return colorsys.rgb_to_hsv(*rgb)


def encode_color(rgb):
    if rgb is None:
        return ""
    return "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in rgb)


def decode_color(text):
    """Inverse of encode_color; None for empty or malformed input (the value
    arrives from other players' packets, so it isn't trusted)."""
    if not isinstance(text, str) or len(text) != 6:
        return None
    try:
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None
