"""
The game mode registry: every GameMode subclass in this package, by id.

Drop a new game mode file in Modules/GameModes/ (a subclass of GameMode -
see deathmatch.py) and it is found automatically: no list to edit - ONLY
WHEN RUNNING FROM SOURCE.

*** ADDING A NEW GAME MODE? You ALSO need one line in                      ***
*** Modules/GameModes/__init__.py: "from .yourfile import YourGameMode" -  ***
*** see its own comment for exactly why. Skip it and the mode works fine   ***
*** from source but SILENTLY VANISHES from a built .exe - this is the      ***
*** EXACT same gotcha that already bit this project once for weapons (see  ***
*** Modules/Weapons/registry.py's own module docstring - the Pencil weapon ***
*** silently vanished from a build before that project added its own       ***
*** version of this same required import).                                ***

The discovery below (pkgutil.iter_modules over this package's own __path__, real
files on a real disk) has no equivalent once Nuitka/PyInstaller has compiled
everything into one binary - there's no directory left to list. Nuitka decides
what to even INCLUDE in that binary by statically tracing real "import"
statements starting from app.py; a module ONLY ever reached by a name computed
at runtime (exactly what the loop below does) is invisible to that trace and
gets silently left out of the build entirely. A new game mode needs a direct
import in __init__.py for exactly this reason, or it will build "successfully"
and just be missing from the mode selector with nothing in the build log to
point at why.

A game mode's id is its `game_mode_id` class attribute, or else its class
name in snake_case ("CheeseWar" -> "cheese_war") - this is what the lobby-
hosting UI's own mode dropdown (Modules/UI/lobby_menu.py) names it by, and
what gets written into Steam lobby metadata so a joining player ends up in
the same mode as the host (see NetworkManager.host_lobby/_lobby_info).
"""

import importlib
import pkgutil
import re

from .game_mode import GameMode

_NOT_GAME_MODES = {"game_mode", "registry"}
_registry = None


def _snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def game_mode_id_of(cls):
    return cls.__dict__.get("game_mode_id") or _snake(cls.__name__)


def _subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _discover():
    global _registry
    if _registry is not None:
        return
    package_module = importlib.import_module(__name__.rsplit(".", 1)[0])
    package = package_module.__name__
    for info in pkgutil.iter_modules(package_module.__path__):
        if info.name not in _NOT_GAME_MODES:
            importlib.import_module(f"{package}.{info.name}")
    _registry = {}
    for cls in _subclasses(GameMode):
        if cls.__dict__.get("abstract"):
            continue
        _registry[game_mode_id_of(cls)] = cls


# The maps each game mode can be played on, by map key (the keys of app.py's MAPS). A mode not listed
# here can be played on every map. The lobby only offers a mode's own maps.
MODE_MAPS = {
    "deathmatch": ("mainmap", "sewer"),
    "cheese_war": ("mainmap",),
}


def mode_allows_map(mode_id, map_key):
    allowed = MODE_MAPS.get(mode_id)
    return allowed is None or map_key in allowed


def coerce_map(mode_id, map_key):
    """map_key if the mode can be played on it, else the mode's first map."""
    return map_key if mode_allows_map(mode_id, map_key) else MODE_MAPS[mode_id][0]


def game_mode_ids():
    _discover()
    return list(_registry)


def get_game_mode_class(game_mode_id):
    """The game mode class with that id, or None if there isn't one."""
    _discover()
    return _registry.get(game_mode_id)


def create_game_mode(game_mode_id, net_mgr, scene=None, spawn_points=None):
    """A new instance of the game mode with that id - or None if
    game_mode_id isn't a registered mode (a typo'd/stale lobby metadata
    value). Every registered mode's own __init__ accepts at least these
    three positional args (GameMode's own base signature - see its module
    docstring); a mode with extra constructor kwargs of its own (like
    Deathmatch's kill_limit) just uses their own defaults here."""
    cls = get_game_mode_class(game_mode_id)
    return cls(net_mgr, scene, spawn_points) if cls is not None else None
