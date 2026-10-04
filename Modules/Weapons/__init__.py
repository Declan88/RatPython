from .weapons_base import WeaponsBase, FireMode
from .usp import USP
from .gouda_gun import GoudaGun
# A weapon module not imported somewhere STATICALLY (a literal "from .x import Y", not
# just registry.py's own dynamic pkgutil.iter_modules/importlib.import_module discovery
# loop) is invisible to Nuitka's (and PyInstaller's) build-time import trace - it starts
# from app.py's own real import statements and has no way to know a module ONLY ever
# reached by a name computed at runtime needs to be compiled in at all, so it's silently
# left out of the built exe entirely. Confirmed as the actual cause of Pencil vanishing
# from a built exe's inventory despite working fine from source (where pkgutil can
# enumerate real files on disk) - registry.py's own "drop a file in, it's found
# automatically" docstring is only true running from source; a frozen build also needs
# a plain import of it from SOMEWHERE, same as USP/GoudaGun already get here (for
# Modules.Weapons's own __all__/convenience access, but it doubles as exactly the static
# reference Nuitka needs). Harmless alongside registry.py's own dynamic import of the
# same module - that just finds this one already loaded.
from .pencil import Pencil

from .inventory import Inventory
from .registry import create_weapon, get_weapon_class, weapon_ids

__all__ = ["WeaponsBase", "FireMode", "USP", "GoudaGun", "Pencil", "Inventory",
           "create_weapon", "get_weapon_class", "weapon_ids"]
