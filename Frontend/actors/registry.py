"""
The actor registry: every ActorBase subclass in this package, by id.

Drop a new actor file in Modules/Actors/ (a subclass of ActorBase - see
cheesepile.py) and it is found automatically: no list to edit - ONLY WHEN
RUNNING FROM SOURCE.

*** ADDING A NEW ACTOR? You ALSO need one line in Modules/Actors/__init__.py: ***
*** "from .yourfile import YourActorClass" - see its own comment for exactly ***
*** why. Skip it and the actor works fine from source but SILENTLY VANISHES ***
*** from a built .exe - this is the EXACT same gotcha that already bit this ***
*** project once for weapons (see Modules/Weapons/registry.py's own module  ***
*** docstring - the Pencil weapon silently vanished from a build before that***
*** project added its own version of this same required import).           ***

The discovery below (pkgutil.iter_modules over this package's own __path__, real
files on a real disk) has no equivalent once Nuitka/PyInstaller has compiled
everything into one binary - there's no directory left to list. Nuitka decides
what to even INCLUDE in that binary by statically tracing real "import"
statements starting from app.py; a module ONLY ever reached by a name computed
at runtime (exactly what the loop below does) is invisible to that trace and
gets silently left out of the build entirely - at runtime the dynamic import
below then finds nothing, and this registry just quietly has one less actor in
it, with no error anywhere. A new actor needs a direct import in __init__.py
for exactly this reason, or it will build "successfully" and just be missing
in-game with nothing in the build log to point at why.

An actor's id is its `actor_id` class attribute, or else its class name in
snake_case ("CheesePile" -> "cheese_pile") - this is what a map's own
"actor_<id>" marker node names it by (see Modules/Graphics/gltf_actors.py).
"""

import importlib
import pkgutil
import re

from .actor_base import ActorBase

_NOT_ACTORS = {"actor_base", "registry"}
_registry = None


def _snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def actor_id_of(cls):
    return cls.__dict__.get("actor_id") or _snake(cls.__name__)


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
        if info.name not in _NOT_ACTORS:
            importlib.import_module(f"{package}.{info.name}")
    _registry = {}
    for cls in _subclasses(ActorBase):
        if cls.__dict__.get("abstract"):
            continue
        _registry[actor_id_of(cls)] = cls


_extra = {}


def register_actor(cls, actor_id=None):
    """Registers an ActorBase subclass defined OUTSIDE this package (a game
    using the Frontend has its own actors) under actor_id, or its own
    actor_id / snake_case class name. Call before loading a map whose
    "actor_<id>" markers should spawn it. Returns cls, so it also works as a
    decorator."""
    _extra[actor_id or actor_id_of(cls)] = cls
    return cls


def actor_ids():
    _discover()
    return list(_registry) + [i for i in _extra if i not in _registry]


def get_actor_class(actor_id):
    """The actor class with that id, or None if there isn't one."""
    _discover()
    return _extra.get(actor_id) or _registry.get(actor_id)


def create_actor(actor_id, scene, position, rotation, scale):
    """A new instance of the actor with that id, spawned at (position,
    rotation, scale) - or None if actor_id isn't a registered actor type
    (a typo'd map marker, or one referring to an actor type that's since
    been removed from the codebase)."""
    cls = get_actor_class(actor_id)
    return cls(scene, position, rotation, scale) if cls is not None else None
