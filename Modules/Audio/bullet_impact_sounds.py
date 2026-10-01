"""Maps a physical material name (Scene.add_static's physical_material,
same names footstep_materials.py already uses) to the set of bullet-
impact .wav variants for it in Assets/Audio/Effects/Impacts, built once
from whatever files actually exist there (each filename is a sound-set
name followed by "_impact_bullet" and a variant digit -
concrete_impact_bullet1.wav, wood_solid_impact_bullet4.wav, ...) rather
than a hardcoded file-list table, so adding/removing a numbered variant
on disk just works with no code change - same approach as
footstep_materials.py's own _build_footstep_index.

The sound SETS on disk (concrete, tile, sand, wood_box, wood_solid,
metal_box, metal_computer, metal_sheet, metal_solid, underwater, ...)
don't line up one-to-one with this project's own physical_material
names (there's no "chainlink" or "dirt" set, for instance - this is a
generic bullet-impact SFX pack, not authored for this exact material
list), so PHYSICAL_MATERIAL_TO_SOUND_SET below bridges the two,
picking the closest-sounding available set for every material that
doesn't have an exact-name match."""

import random
import re
from pathlib import Path

_SOUND_DIR = Path("Assets/Audio/Effects/Impacts")
_FILENAME_RE = re.compile(r"^(.+)_impact_bullet\d+$", re.IGNORECASE)

DEFAULT_BULLET_IMPACT_SOUND_SET = "concrete"

# physical_material (Scene.add_static/footstep_materials.py's own names)
# -> which sound set under Assets/Audio/Effects/Impacts to play for it.
# Exact matches (concrete, sand, tile) just name themselves; everything
# else picks the nearest available set by ear:
#   - chainlink/duct/metal/metalgrate: all thin sheet metal in practice
#     (see surfaceproperties.txt's own "metal" comment - "almost nothing
#     is solid metal") -> metal_sheet.
#   - ladder: solid metal rungs/rails, not sheet -> metal_solid.
#   - dirt/grass/gravel/mud/snow: no earthy set exists at all in this
#     pack - sand's soft, low-pitched thud is the closest stand-in for
#     all of them.
#   - wade/slosh: both water - underwater is the obvious match.
#   - wood: generic/solid wood (trees, benches) -> wood_solid.
#   - woodpanel: thinner paneling/crate-like resonance -> wood_box.
PHYSICAL_MATERIAL_TO_SOUND_SET = {
    "concrete": "concrete",
    "tile": "tile",
    "sand": "sand",
    "chainlink": "metal_sheet",
    "duct": "metal_sheet",
    "metal": "metal_sheet",
    "metalgrate": "metal_sheet",
    "ladder": "metal_solid",
    "dirt": "sand",
    "grass": "sand",
    "gravel": "sand",
    "mud": "sand",
    "snow": "sand",
    "wade": "underwater",
    "slosh": "underwater",
    "wood": "wood_solid",
    "woodpanel": "wood_box",
}


def _build_sound_index():
    index = {}
    if not _SOUND_DIR.is_dir():
        return index
    for path in sorted(_SOUND_DIR.iterdir()):
        if path.suffix.lower() != ".wav":
            continue
        match = _FILENAME_RE.match(path.stem)
        if not match:
            continue
        sound_set = match.group(1).lower()
        index.setdefault(sound_set, []).append(str(path))
    return index


# Scanned once at import time - this folder is fixed game content, same
# reasoning as footstep_materials.py's own FOOTSTEP_SOUNDS.
BULLET_IMPACT_SOUNDS = _build_sound_index()


def get_bullet_impact_sound(material):
    """Returns a random file path for `material`'s mapped sound set
    (case-insensitive), falling back to DEFAULT_BULLET_IMPACT_SOUND_SET
    if `material` is None, unrecognized, or maps to a set with no files.
    Returns None only if even the default has no files."""
    key = (material or "").lower()
    sound_set = PHYSICAL_MATERIAL_TO_SOUND_SET.get(key, DEFAULT_BULLET_IMPACT_SOUND_SET)
    choices = BULLET_IMPACT_SOUNDS.get(sound_set) or BULLET_IMPACT_SOUNDS.get(DEFAULT_BULLET_IMPACT_SOUND_SET)
    if not choices:
        return None
    return random.choice(choices)
