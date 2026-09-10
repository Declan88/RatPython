"""Maps a physical material name (e.g. "dirt", "concrete") to the set
of footstep .wav variants for it in Assets/Audio/footsteps, built once
from whatever files actually exist there (each Source-style filename is
a material name followed by a variant digit - dirt1.wav, dirt2.wav,
... snow6.wav) rather than a hardcoded material -> file-list table, so
adding/removing a numbered variant on disk just works with no code
change.

DEFAULT_FOOTSTEP_MATERIAL is the fallback used both when a walkable
surface was never given a physical_material (see Scene.add_static/
add_dynamic) and when one was given but doesn't match any folder of
actual files (a typo, or a material with no recorded sounds yet) -
better to play a plausible generic footstep than none at all.
"""

import random
import re
from pathlib import Path

_FOOTSTEP_DIR = Path("Assets/Audio/footsteps")
_FILENAME_RE = re.compile(r"^([a-zA-Z]+?)(\d+)$")

DEFAULT_FOOTSTEP_MATERIAL = "concrete"


def _build_footstep_index():
    index = {}
    if not _FOOTSTEP_DIR.is_dir():
        return index
    for path in sorted(_FOOTSTEP_DIR.iterdir()):
        if path.suffix.lower() != ".wav":
            continue
        match = _FILENAME_RE.match(path.stem)
        if not match:
            continue
        material = match.group(1).lower()
        index.setdefault(material, []).append(str(path))
    return index


# Scanned once at import time - this folder is fixed game content, not
# something that changes while the process is running.
FOOTSTEP_SOUNDS = _build_footstep_index()


def get_footstep_sound(material):
    """Returns a random file path for `material` (case-insensitive),
    falling back to DEFAULT_FOOTSTEP_MATERIAL if `material` is None or
    unrecognized. Returns None only if even the default has no files
    (e.g. the footsteps folder is missing entirely)."""
    key = (material or DEFAULT_FOOTSTEP_MATERIAL).lower()
    choices = FOOTSTEP_SOUNDS.get(key) or FOOTSTEP_SOUNDS.get(DEFAULT_FOOTSTEP_MATERIAL)
    if not choices:
        return None
    return random.choice(choices)


# Per-surface (walk_volume, run_volume) pairs, read directly out of
# Source SDK 2013's CBasePlayer::UpdateStepSound (game/shared/
# baseplayer_shared.cpp) rather than approximated - that function's own
# switch on psurface->game.material only special-cases 3 groups
# (CHAR_TEX_DIRT, CHAR_TEX_VENT, plus the ladder/water-knee paths
# outside the switch entirely); CHAR_TEX_CONCRETE, _METAL, _GRATE,
# _TILE, _SLOSH, and every material Source itself doesn't list are all
# numerically identical to `default` there (0.2/0.5) - not omissions on
# this port's part, that really is what Source's own code does. Ladder
# and wade (water-knee) are each a single fixed fvol in the original
# (0.5 and 0.65) rather than a walk/run pair, kept as equal tuples here
# so callers don't need a separate code path for them.
_MATERIAL_VOLUME = {
    "dirt": (0.25, 0.55),
    "mud": (0.25, 0.55),      # not in Source's own switch (falls to
                               # default there) but close enough in
                               # kind to dirt that reusing dirt's pair
                               # reads better than the generic default
    "duct": (0.4, 0.7),       # CHAR_TEX_VENT
    "ladder": (0.5, 0.5),     # fixed fvol=0.5, not walk/run split
    "wade": (0.65, 0.65),     # water-knee path, fixed fvol=0.65
}

# CHAR_TEX_CONCRETE/_METAL/_GRATE/_TILE/_SLOSH and Source's own
# `default:` case - also water-feet (bWalking ? 0.2 : 0.5, numerically
# identical). Applies to concrete, metal, metalgrate, tile, slosh,
# grass, gravel, sand, snow, wood, woodpanel, chainlink - none of which
# Source's own switch calls out separately.
_DEFAULT_VOLUME = (0.2, 0.5)


def get_footstep_volume(material, walking):
    """Returns Source's own fvol for `material` (see UpdateStepSound) -
    walking=True selects the slower/quieter of the pair, walking=False
    the faster/louder one (matches UpdateStepSound's own `bWalking`).
    Ducking's extra "fvol *= 0.65" is NOT applied here - that's a
    separate, stance-based multiplier orthogonal to which surface is
    underfoot, so callers apply it themselves on top of this."""
    pair = _MATERIAL_VOLUME.get((material or "").lower(), _DEFAULT_VOLUME)
    return pair[0] if walking else pair[1]
