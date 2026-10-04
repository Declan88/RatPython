"""
Base class for a game mode: whatever, beyond raw movement/shooting, decides
where a player (re)spawns, how a kill counts, and when a round ends.

Kill/death COUNTS themselves already live on NetworkManager (the local
player: see its local_kills/local_deaths properties) and RemotePlayer (every
other player: .kills/.deaths, replicated from their own NetworkManager over
the wire - see NetworkManager's own "ki"/"x" packet fields and the
kill-credit message it sends). A GameMode does NOT keep a second copy of
that - it reads it (see standings() below), so score is always exactly what
every peer's own client already agrees happened, regardless of which mode is
active or when a client joined. A GameMode's own job is everything ELSE a
"mode" traditionally decides: spawn selection, and hooks a subclass can use
for mode-specific rules (a win condition, a score multiplier, a match
announcement) - see deathmatch.py for a concrete example.

app.py owns exactly one GameMode instance at a time (created alongside
NetworkManager) and calls into it wherever the old code had a mode-shaped
decision hardcoded - see app.py's own game_mode.choose_spawn_position() call
sites (replacing a single hardcoded SPAWN_POSITION) and its net_mgr.on_kill
wiring.
"""


class GameMode:
    """name is a short display label a future mode-select UI could show -
    purely cosmetic, nothing here reads it."""
    name = "Game Mode"

    def __init__(self, net_mgr, scene=None, spawn_points=None):
        self.net_mgr = net_mgr
        # Set by app.py whenever the active map changes (a mode that wants
        # to reason about the level - e.g. pick a spawn point via a raycast,
        # or read map-authored spawn markers once that exists - has it
        # available; the base class and Deathmatch below don't need it).
        self.scene = scene
        # [(x, y, z), ...] candidate spawn points, hull-centre height, world
        # units. No spawn_points configured means "spawn everyone at this
        # one point", which is exactly today's single-SPAWN_POSITION
        # behavior - see choose_spawn_position.
        self.spawn_points = list(spawn_points) if spawn_points else [(0.0, 2.0, 3.0)]

    def choose_spawn_position(self):
        """Where the local player (re)spawns - called by app.py's
        setup_game/start_game (reuse path)/end_death, i.e. every point that
        used to just read the SPAWN_POSITION constant directly. Base
        implementation: always the first configured spawn point (or the
        single default above) - a fixed spawn, deliberately the simplest
        possible mode so a caller with no real mode configured yet still
        gets consistent, correct behavior. Override for anything smarter
        (random choice, avoiding other players - see Deathmatch)."""
        return self.spawn_points[0]

    def choose_spawn_rotation(self):
        """(yaw, pitch) degrees the camera snaps to on every (re)spawn - same
        call sites as choose_spawn_position (app.py's setup_game/start_game's
        reuse path/end_death), always applied together with it. A FIXED value
        deliberately - a respawning player's camera used to just keep
        whatever orientation it already had (frozen from the death animation,
        or carried over from however the menu camera happened to be looking),
        which let a player choose which way they'd be facing the instant they
        respawned by choosing where to look right before dying. Spawning
        looking a known, fixed direction removes that - nobody can set up
        their own respawn facing. -90.0 matches Camera's own default yaw (see
        Modules/Camera/camera.py) purely so a spawn with no real mode
        configured yet looks the same direction Camera already starts facing
        before any match begins, not for any deeper reason. Override for a
        mode that wants each spawn POINT to face a particular way (a per-
        point yaw stored alongside spawn_points, say) instead of one fixed
        direction for every spawn."""
        return -90.0, 0.0

    def on_kill(self, killer_id, victim_id, weapon_name=""):
        """Called once for every kill THIS client has full information for -
        in practice that's only kills the local player scored (app.py wires
        NetworkManager.on_kill to this, killer_id always == net_mgr.
        local_steam_id in that case) - a client only directly learns of its
        OWN kills; it observes everyone else's kills/deaths purely through
        their already-replicated counters (see standings()), same as any
        other multiplayer game's remote scoreboard. Flat 1-point-per-kill
        scoring is already the wire protocol's own default (see module
        docstring) - this hook exists for anything ON TOP of that a mode
        wants: an announcement, a win-condition check (see is_match_over),
        a streak bonus. No-op by default."""

    def is_match_over(self):
        """Whichever mode-specific condition ends the round (a kill/score
        limit, a timer, ...). False (never) by default - a mode with no
        target plays indefinitely, matching every match today before this
        existed. Not currently polled anywhere in app.py - wiring a real
        end-of-match screen to this is future work; it's here now so a
        mode CAN express the condition without app.py needing changes
        later to make room for it."""
        return False

    def standings(self):
        """[(steam_id, name, kills, deaths), ...], best first (most kills,
        then fewest deaths as the tiebreaker) - built fresh from
        NetworkManager's own replicated counters every call (there are only
        ever a handful of players, so this is cheap enough to not cache).
        Scoreboard.py calls this rather than reading net_mgr/remote_players
        directly, so a future mode with fundamentally different scoring
        (capture points, objective time...) can override just this one
        method and the scoreboard follows automatically without its own
        code needing to know which mode is active."""
        net = self.net_mgr
        rows = [(net.local_steam_id, net.local_name or "You", net.local_kills, net.local_deaths)]
        for steam_id, player in net.remote_players.items():
            rows.append((steam_id, player.name or f"Player {steam_id % 10000}", player.kills, player.deaths))
        rows.sort(key=lambda row: (-row[2], row[3]))
        return rows
