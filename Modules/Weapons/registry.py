"""
The weapon registry: every WeaponsBase subclass in this package, by id.

Drop a new weapon file in Modules/Weapons/ (a subclass of WeaponsBase - see usp.py)
and it is found automatically: no list to edit. A weapon's id is its `weapon_id`
class attribute, or else its class name in snake_case ("GoudaGun" -> "gouda_gun").
The id is what goes over the network to tell other players which weapon is in
someone's hands, and what an Inventory / loadout names a weapon by.
"""

import importlib
import pkgutil
import re

from .weapons_base import WeaponsBase

_NOT_WEAPONS = {"weapons_base", "recoil", "explosion", "registry", "inventory"}
_registry = None


def _snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def weapon_id_of(cls):
    return cls.__dict__.get("weapon_id") or _snake(cls.__name__)


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
        if info.name not in _NOT_WEAPONS:
            importlib.import_module(f"{package}.{info.name}")
    _registry = {}
    for cls in _subclasses(WeaponsBase):
        if cls.__dict__.get("abstract"):
            continue
        _registry[weapon_id_of(cls)] = cls


def weapon_ids():
    _discover()
    return list(_registry)


def get_weapon_class(weapon_id):
    """The weapon class with that id, or None if there isn't one."""
    _discover()
    return _registry.get(weapon_id)


def create_weapon(weapon_id):
    """A new instance of the weapon with that id, or None if unknown."""
    cls = get_weapon_class(weapon_id)
    return cls() if cls is not None else None
