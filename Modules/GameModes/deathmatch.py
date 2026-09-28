"""
Standard free-for-all: everyone against everyone, a flat point per kill
(already the wire protocol's own default - see GameMode's module docstring),
spawning at a random configured point that's clear of any other player
currently standing near one (the classic "don't spawn in someone's
crosshair" rule), and an optional kill-limit win condition.
"""

import random

import glm

from .game_mode import GameMode

# A spawn point with an enemy at least this close (metres) is skipped in
# favor of a farther one, whenever a farther choice actually exists.
MIN_SPAWN_CLEARANCE = 6.0


class Deathmatch(GameMode):
    name = "Deathmatch"

    def __init__(self, net_mgr, scene=None, spawn_points=None, kill_limit=0):
        super().__init__(net_mgr, scene, spawn_points)
        # 0 = no limit - a session just tracks standings forever and never
        # calls itself over on its own (see is_match_over); app.py's own
        # disconnect/ESC flow still ends a match at any time regardless,
        # exactly like every match before game modes existed.
        self.kill_limit = kill_limit

    def choose_spawn_position(self):
        """A random configured spawn point, preferring one far from every
        other player's last-known position over one right next to someone.
        Falls back to GameMode's own single-point behavior if only one spawn
        point is configured (nothing to choose between)."""
        points = self.spawn_points
        if len(points) <= 1:
            return points[0]

        others = [
            player.model.obj["position"] for player in self.net_mgr.remote_players.values()
            if player.model.obj is not None
        ]

        def clearance(point):
            if not others:
                return float("inf")
            pv = glm.vec3(*point)
            return min(glm.distance(pv, other) for other in others)

        clear_enough = [p for p in points if clearance(p) >= MIN_SPAWN_CLEARANCE]
        return random.choice(clear_enough or points)

    def on_kill(self, killer_id, victim_id, weapon_name=""):
        pass  # flat 1-point-per-kill is already the wire default - nothing extra to add

    def is_match_over(self):
        if self.kill_limit <= 0:
            return False
        return any(kills >= self.kill_limit for _, _, kills, _deaths in self.standings())
