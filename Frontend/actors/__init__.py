"""
Map actors: dynamic props a level's "actor_<id>" marker nodes spawn (see
actor_base.py). The Frontend ships none of its own - subclass ActorBase in
your game and call register_actor() (see registry.py).
"""

from .actor_base import ActorBase
from .registry import actor_ids, create_actor, get_actor_class, register_actor
