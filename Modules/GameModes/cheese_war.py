"""
Cheese War - a second game mode, distinct from Deathmatch mainly so there's
a real second entry in the mode registry/selector (Modules/GameModes/
registry.py, the lobby-hosting UI's own mode dropdown) and so Modules/
Actors/cheesepile.py's own CheesePile actor has something to check against:
it only stays spawned while THIS mode is active (see its own update()
override), despawning itself in any other mode.

Subclasses Deathmatch outright rather than GameMode directly - same flat
point-per-kill scoring and "don't spawn next to an enemy" spawn-clearance
logic apply here too.

The cheese economy: there is one universal supply of CHEESE_TOTAL. A player
standing inside the cheese pile very slowly drains it into their OWN cheese
value (a scoreboard column, replicated to everyone like kills/deaths - see
GameMode.set_local_stat); the pile shrinks permanently as they do. Each
client is authoritative only for its own cheese; the pile is never sent
anywhere, it is DERIVED as CHEESE_TOTAL minus everyone's cheese - so every
peer agrees on it without any extra sync, and a player who leaves returns
their cheese to it.
"""

import math

from .deathmatch import Deathmatch

CHEESE_TOTAL = 500.0       # the universal supply: pile + all players' cheese
PULL_RATE = 3.0           # cheese/second drained from the pile into a player standing in it
CHEESE_KEY = "cheese"


class CheeseWar(Deathmatch):
    name = "Cheese War"
    game_mode_id = "cheese_war"
    stat_columns = ((CHEESE_KEY, "Cheese"),)

    def __init__(self, net_mgr, scene=None, spawn_points=None, kill_limit=0):
        super().__init__(net_mgr, scene, spawn_points, kill_limit)
        self.set_local_stat(CHEESE_KEY, 0.0)
        # {pile actor id: set of player ids inside it this frame} - see report_pile_occupants.
        self._occupants = {}

    def choose_spawn_rotation(self):
        """Spawn looking at the nearest cheese pile (level, no pitch) - the fixed facing
        otherwise (see GameMode.choose_spawn_rotation for why it's never player-chosen)."""
        piles = [a for a in getattr(self.scene, "actors", ()) if getattr(a, "actor_id", None) == "cheesepile"]
        if not piles:
            return super().choose_spawn_rotation()
        x, _y, z = self.last_spawn_position
        pile = min(piles, key=lambda a: (a.position.x - x) ** 2 + (a.position.z - z) ** 2).position
        dx, dz = pile.x - x, pile.z - z
        if dx == 0.0 and dz == 0.0:
            return super().choose_spawn_rotation()
        return math.degrees(math.atan2(dz, dx)), 0.0

    def report_pile_occupants(self, pile_key, player_ids):
        """Called every frame by each CheesePile actor with the ids of whoever is
        standing in it (the local player's own id included, only while alive)."""
        self._occupants[pile_key] = set(player_ids)

    def clear_pile(self, pile_key):
        self._occupants.pop(pile_key, None)

    @property
    def cheese_in_play(self):
        """Total cheese currently held by players (local + every remote)."""
        net = self.net_mgr
        total = net.local_mode_values.get(CHEESE_KEY, 0.0)
        for player in net.remote_players.values():
            total += player.mode_values.get(CHEESE_KEY, 0.0)
        return total

    @property
    def pile_value(self):
        """What's left in the pile: CHEESE_TOTAL minus everyone's cheese."""
        return max(0.0, min(CHEESE_TOTAL, CHEESE_TOTAL - self.cheese_in_play))

    @property
    def pile_fraction(self):
        """pile_value as 0..1 of CHEESE_TOTAL (1 = full)."""
        return self.pile_value / CHEESE_TOTAL

    def update(self, dt):
        super().update(dt)
        local_id = self.net_mgr.local_steam_id
        mine = self.net_mgr.local_mode_values.get(CHEESE_KEY, 0.0)
        if any(local_id in ids for ids in self._occupants.values()):
            mine += min(PULL_RATE * dt, self.pile_value)
        self.set_local_stat(CHEESE_KEY, mine)
