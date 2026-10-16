"""
Standard free-for-all: everyone against everyone, a flat point per kill
(already the wire protocol's own default - see GameMode's module docstring),
spawning away from other players (GameMode.choose_spawn_position), and an
optional kill-limit win condition.
"""

from .game_mode import GameMode


class Deathmatch(GameMode):
    name = "Deathmatch"
    round_duration = 15 * 60  # 15 minutes

    def __init__(self, net_mgr, scene=None, spawn_points=None, kill_limit=0):
        super().__init__(net_mgr, scene, spawn_points)
        # 0 = no limit - a session just tracks standings forever and never
        # calls itself over on its own (see is_match_over); app.py's own
        # disconnect/ESC flow still ends a match at any time regardless,
        # exactly like every match before game modes existed.
        self.kill_limit = kill_limit

    def on_kill(self, killer_id, victim_id, weapon_name=""):
        pass  # flat 1-point-per-kill is already the wire default - nothing extra to add

    def is_match_over(self):
        if self.kill_limit <= 0:
            return False
        return any(kills >= self.kill_limit for _, _, kills, _deaths in self.standings())
