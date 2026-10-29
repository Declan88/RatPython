"""
*** ADDING A NEW GAME MODE? You need a line below, "from .yourfile import  ***
*** YourGameMode" - see registry.py's own module docstring for exactly why ***
*** this is required, not optional, once this project is built with       ***
*** Nuitka/PyInstaller (registry.py's own pkgutil-based auto-discovery is  ***
*** invisible to that build's static-import trace - this exact gotcha      ***
*** already bit the Pencil weapon once, see Modules/Weapons/registry.py). ***
"""

from .game_mode import GameMode
from .deathmatch import Deathmatch
from .cheese_war import CheeseWar

from .registry import create_game_mode, get_game_mode_class, game_mode_ids, mode_allows_map, coerce_map

__all__ = [
    "GameMode", "Deathmatch", "CheeseWar",
    "create_game_mode", "get_game_mode_class", "game_mode_ids", "mode_allows_map", "coerce_map",
]
