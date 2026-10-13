"""
Map actors - see actor_base.py's own module docstring for what these are.

*** ADDING A NEW ACTOR? You need a line below, "from .yourfile import       ***
*** YourActorClass" - see registry.py's own module docstring for exactly   ***
*** why this is required, not optional, once this project is built with    ***
*** Nuitka/PyInstaller (registry.py's own pkgutil-based auto-discovery is  ***
*** invisible to that build's static-import trace - this exact gotcha      ***
*** already bit the Pencil weapon once, see Modules/Weapons/registry.py).  ***
"""

from .actor_base import ActorBase
from .cheesepile import CheesePile

from .registry import create_actor, get_actor_class, actor_ids
