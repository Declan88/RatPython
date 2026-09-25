"""
The game's UI palette - cool blue accents over deep night blues.
Colors are 0-255 RGBA tuples like every widget takes.
"""

ACCENT = (96, 165, 250, 255)         # bright sky blue
ACCENT_DIM = (58, 100, 160, 255)
FUR = (170, 138, 112, 255)
TEXT = (240, 234, 230, 255)
MUTED = (160, 154, 166, 255)
WARN = (255, 210, 120, 255)

SIDEBAR_WIDTH = 900                  # logical px; the 3D preview lives to its right
SIDEBAR = (9, 11, 19, 225)           # full-height menu column
CARD = (18, 21, 33, 235)             # panels sitting on it / floating cards
CARD_BORDER = (52, 62, 84, 255)

BUTTON = dict(color=(34, 38, 56, 240), hover_color=(48, 82, 140, 250),
              pressed_color=(32, 54, 96, 255))


# ---- fonts -------------------------------------------------------------
# Typefaces are looked up in the OS font folders (pygame's bundled font is the
# fallback when none is found): body text uses a bold-ish UI face, the title an
# extra-heavy one.
_BODY_FONTS = ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf")
_TITLE_FONTS = ("seguibl.ttf", "ariblk.ttf", "arialbd.ttf")
_font_cache = {}


def _find_font(candidates):
    import os
    folders = (os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
               "/usr/share/fonts/truetype/msttcorefonts", "/Library/Fonts")
    for folder in folders:
        for name in candidates:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    return None


def font_path():
    """Default UI typeface (None -> pygame's bundled font)."""
    if "body" not in _font_cache:
        _font_cache["body"] = _find_font(_BODY_FONTS)
    return _font_cache["body"]


def title_font_path():
    """Heavy weight for the game title."""
    if "title" not in _font_cache:
        _font_cache["title"] = _find_font(_TITLE_FONTS)
    return _font_cache["title"]
