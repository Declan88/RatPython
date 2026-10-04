import hashlib
import json
import secrets
import time

# steam_api64.dll is already pre-loaded globally (ctypes.RTLD_GLOBAL) by
# app.py, BEFORE this module is ever imported - app.py computes that
# path relative to itself (the project root), so it's portable across
# machines. This file used to duplicate that loading logic with its own
# hardcoded path search, including a hardcoded absolute path specific to
# one developer's machine (E:\Python\RatWar\steam_api64.dll) that
# doesn't exist anywhere else - on any other machine, all its candidate
# paths failed and it called sys.exit(1), crashing the whole app despite
# app.py having already loaded the DLL correctly moments earlier. Removed
# entirely rather than fixed, since it was never needed in the first place.

import py_steam_net
from .remote_player import RemotePlayer
from Modules.Player.rat_colors import encode_color


class NetworkManager:
    """
    Ported from a Panda3D-based version. Two things had to change:

    1. The original used Panda3D's global taskMgr (from
       direct.task.TaskManagerGlobal) to poll network callbacks and
       broadcast the local transform every frame, plus
       taskMgr.doMethodLater() for delayed roster printing. This
       project has no Panda3D event loop anywhere - app.py runs its
       own plain `while running:` loop - so those registered tasks
       would never actually run. Replaced with a single update()
       method you call once per frame yourself (see app.py), and a
       tiny internal deferred-call list (_schedule/_run_deferred)
       using wall-clock time instead of taskMgr.doMethodLater.

    2. The original read the local player's transform off a Panda3D
       NodePath (player_node.getPos()/getHpr()). This engine's Camera
       class exposes a glm.vec3 .position plus separate .yaw/.pitch
       floats instead - there's no NodePath, so this now takes the
       Camera directly rather than an "app" object with a ".player".

    `scene` is threaded through for the same reason as `camera` above -
    RemotePlayer needs it to build a visible PlayerModel via
    scene.add_skeletal/scene.physics, and there's no Panda3D-style
    global scene graph it could reach that through on its own.
    """

    # Movement packets: at most this often, plus a heartbeat when idle.
    SEND_INTERVAL = 1.0 / 30.0
    RECEIVE_BATCH = 256  # max packets read per frame - drains a backlog after a map load
    HEARTBEAT_INTERVAL = 0.25

    def __init__(self, camera, scene=None):
        self.camera = camera
        # None while in menus; app.py sets it (and in_game) once a map is
        # loaded - RemotePlayer needs the scene, and there's nothing to
        # broadcast before then.
        self.scene = scene
        self.in_game = False
        self.remote_players = {}
        self.current_lobby_id = None
        self._local_state = None
        self.local_hat = None   # short hat name from the main menu, or None
        self.local_weapon = None   # id (Modules/Weapons/registry.py) of the weapon in our hands, sent so others draw it
        self.local_color = None  # (r, g, b) 0-1 fur color from the main menu, or None
        self._shot_count = 0    # shots the local player has fired (sent as a counter, like jumps)
        self._footstep_count = 0   # local footsteps (a counter too, like shots/jumps)
        self._last_footstep = ("", 1.0)   # (material, volume) of the latest one
        self._death_count = 0   # times the local player has died (a counter too)
        self._death_push = None   # explosion shove for our death gibs (see notify_death)
        self._death_damage_class = "bullet"   # see notify_death - what killed us last, by name
        self._kill_count = 0    # kills WE'VE been credited for - see notify_death/_send_kill_credit
        # Who last damaged us, and when (perf_counter seconds) - notify_death
        # credits a kill to whoever this points at, if they're still "recent"
        # enough (see KILL_CREDIT_WINDOW), same idea as a real shooter's own
        # last-hit-wins kill attribution.
        self._last_attacker_id = None
        self._last_attacker_time = 0.0
        self._last_attacker_weapon = ""
        self._last_attacker_headshot = False
        self.local_alive = True
        self.profiler = None    # Modules/Debug FrameProfiler, when the game wants network timings
        self._recent_shot_ends = []   # end points of our last few shots (tracers), oldest first
        self.on_death = None    # callback(feet_position, velocity, fur colour, damage_class) when another player dies (gibs, or dissolves for Zap)
        self.on_tracer = None   # callback(start, end, follow, shooter_weapon) to draw another player's tracer, flash and impact
        self.on_footstep = None  # callback(material, position, volume) to play another player's footstep
        self.on_dissolve_spark = None  # callback(position) - a Zap-dissolving remote player's periodic spark/glow burst
        self.on_dissolve_arc = None    # callback(position) - a Zap-dissolving remote player's periodic tesla-arc burst
        self.on_damage = None   # callback(amount, attacker_steam_id, weapon_name, headshot, push, damage_class, origin) when someone shoots us - origin: (x,y,z) or None, see send_damage's own docstring
        # callback(killer_steam_id, victim_steam_id, weapon_name, headshot) for
        # EVERY kill in the match this client learns about (see
        # _broadcast_kill/_receive_killfeed) - not just our own, so a kill
        # feed UI built off this is the same for every player, CS:GO-style.
        self.on_killfeed = None
        self.on_chat = None      # callback(sender_steam_id, text) for a chat message from someone else
        self.on_admin_kill = None  # callback(sender_steam_id) - see send_admin_kill; verify sender is the admin yourself
        self.local_name = ""    # our Steam persona name, sent in every packet
        self.local_steam_id = 0
        self._avatars = {}        # steam id -> 64x64 RGBA bytes, once Steam has them
        self._avatar_tried = {}   # steam id -> last time we asked
        self._jump_count = 0
        self._last_sent_state = None
        self._last_send_time = 0.0
        self._last_update_time = 0.0
        self._last_prune_time = 0.0
        self._members = set()
        self._last_hello_time = 0.0
        self._net_seen = set()
        self._relay_ok = False
        self._member_order = []
        self._retry_at = {}
        self._failed_logged = set()
        self._heard_from = set()  # peers that have sent us anything (incl. hellos)
        self._net_log_t0 = {}   # peer -> first hello time (diagnostics)
        self._members_checked = 0.0
        self._deferred = []  # list of (fire_time, func) - replaces taskMgr.doMethodLater
        self._pending_host = None
        self._pending_join = None

        self._init_steam()

    def _init_steam(self):
        """(Re)connects to Steam - the constructor's own first attempt, and reconnect()'s retry
        (see its own docstring) both go through this so the two can't drift apart. Leaves
        self.client as None (available stays False) on any failure, exactly like the
        constructor always has - callers check that, not this method's return value, except
        reconnect() itself, which does want to know whether THIS attempt worked."""
        try:
            self.client = py_steam_net.PySteamClient()
            self.client.init(480)
            print(f"Steam P2P initialized successfully. Ready: {self.client.is_ready()}")

            self.local_steam_id = self.client.own_steam_id()
            print(f"Your Steam ID: {self.local_steam_id}")
            self.local_name = self.client.own_name() if hasattr(self.client, "own_name") else ""
            print(f"Your Steam name: {self.local_name}")

            self.client.set_message_recv_callback(self.handle_data)
            self.client.set_lobby_changed_callback(self.on_lobby_changed)
            self.client.set_connection_failed_callback(self.on_session_failed)
            return True
        except Exception as e:
            print(f"Steam initialization failed: {e}. Make sure Steam client is running.")
            self.client = None
            return False

    def reconnect(self):
        """Retries Steam initialization from scratch - for a "Reconnect" button in the menu
        when the game started before Steam had (or Steam was closed and reopened since): the
        constructor only ever gets ONE attempt, at launch, so without this there was no way
        back into a Steam-having-players state short of restarting the whole game. Returns
        whether THIS attempt succeeded (self.available reflects it either way)."""
        return self._init_steam()

    @property
    def available(self):
        return self.client is not None

    # =============================================================
    # PER-FRAME UPDATE - call this once per frame from your main loop
    # =============================================================

    SESSION_RETRY_DELAY = 3.0

    def on_session_failed(self, steam_id):
        # Steam gave up on opening a session. Re-sending immediately (the
        # stream ran at 30Hz) opened a NEW connection request every time while
        # the peer was still settling the previous one, which Steam drops
        # ("Symmetric role resolution ... already the server") - so every
        # attempt kept failing. Back off and let the slow hello retry instead.
        self._retry_at[steam_id] = time.perf_counter() + self.SESSION_RETRY_DELAY
        if steam_id not in self._failed_logged:
            self._failed_logged.add(steam_id)
            print(f"[Net] Steam P2P session to {steam_id} failed - retrying every {self.SESSION_RETRY_DELAY:.0f}s")

    def pump_callbacks(self):
        """Runs Steam callbacks only - safe to call from inside a long blocking
        load (see scene_base.LOAD_PUMP)."""
        if self.client:
            self.client.run_callbacks()

    def leave_lobby(self):
        """Leaves the current lobby and forgets every other player (back to the main menu)."""
        if self.client and self.current_lobby_id:
            try:
                self.client.leave_lobby(self.current_lobby_id)
            except Exception as e:
                print(f"[Net] leave_lobby failed: {e}")
        for remote in self.remote_players.values():
            remote.destroy()
        self.remote_players.clear()
        self.current_lobby_id = None
        self.in_game = False
        self._members = set()
        self._member_order = []
        self._heard_from.clear()
        self._net_seen.clear()
        self._retry_at.clear()
        self._failed_logged.clear()
        self._last_sent_state = None
        self._pending_host = self._pending_join = None
        self.local_alive = True

    @property
    def is_host(self):
        """Whether this player owns the lobby (the first member)."""
        return bool(self._member_order) and self._member_order[0] == self.local_steam_id

    def update(self):
        if not self.client:
            return
        prof = self.profiler if self.profiler is not None and self.profiler.enabled else None
        t = time.perf_counter() if prof else 0.0
        self.client.run_callbacks()
        self._run_deferred()
        if self.current_lobby_id:
            self._send_hellos(time.perf_counter())
        if prof:
            t2 = time.perf_counter(); prof.add("  net: callbacks+hellos", t2 - t); t = t2
        if self.in_game:
            # handle_data only ever runs from here: py_steam_net hands over
            # incoming packets when (and only when) they're polled for. Nothing
            # polled before this - other players could join the lobby but their
            # positions were never read, so their models never appeared.
            self.client.receive_messages(0, self.RECEIVE_BATCH)
            now = time.perf_counter()
            if prof:
                prof.add("  net: receive+parse", now - t)
            dt = now - self._last_update_time if self._last_update_time else 0.0
            self._last_update_time = now
            self._broadcast_transform(now)
            if prof:
                t2 = time.perf_counter(); prof.add("  net: send state", t2 - now); t = t2
            for remote in self.remote_players.values():
                remote.update(dt)
            if prof:
                t2 = time.perf_counter(); prof.add("  net: remote players", t2 - t); t = t2
            if now - self._last_prune_time >= 1.0:
                self._last_prune_time = now
                self._prune_remote_players()
                self._refresh_names()
                if prof:
                    prof.add("  net: prune+names (1/s)", time.perf_counter() - t)

    def _schedule(self, delay_seconds, func):
        """Replaces taskMgr.doMethodLater - runs func() once, after at
        least delay_seconds have passed, the next time update() runs."""
        self._deferred.append((time.perf_counter() + delay_seconds, func))

    def _run_deferred(self):
        if not self._deferred:
            return
        now = time.perf_counter()
        remaining = []
        for fire_time, func in self._deferred:
            if now >= fire_time:
                try:
                    func()
                except Exception as e:
                    print(f"[NetworkManager] deferred call failed: {e}")
            else:
                remaining.append((fire_time, func))
        self._deferred = remaining

    # =============================================================
    # LOBBY MANAGEMENT
    # =============================================================

    def print_session_roster(self):
        if not self.current_lobby_id:
            return
        try:
            members = self.client.get_lobby_members(self.current_lobby_id)
            if not members:
                return

            host_id = members[0]
            print("\n===============================")
            print("       SESSION ROSTER          ")
            print("===============================")
            print(f" Active Lobby ID: {self.current_lobby_id}")
            print("-------------------------------")
            for member_id in members:
                role = "HOST" if member_id == host_id else "CLIENT"
                is_local = " (You)" if member_id == self.local_steam_id else ""
                print(f" - Steam ID: {member_id}{is_local} [{role}]")
            print("===============================\n")
        except Exception as e:
            print(f"Error printing session roster: {e}")

    # =============================================================
    # HOST / LIST / JOIN
    #
    # Every call into py_steam_net that takes &mut self (create_lobby,
    # join_lobby, get_lobby_list) goes through _schedule(0, ...) rather
    # than being called directly: these methods are reached from UI
    # events or from Steam callbacks that fire INSIDE run_callbacks(),
    # and a &mut self call while that borrow is outstanding raises PyO3's
    # "Already borrowed". The on_result callbacks below can themselves
    # run inside run_callbacks(), so they must only record state, never
    # call back into self.client.
    # =============================================================

    @staticmethod
    def hash_password(salt, password):
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    @classmethod
    def check_password(cls, lobby_info, password):
        """Client-side check of a listed lobby's password. Steam has no
        lobby-password concept, so the host publishes a salted hash in
        lobby data and joiners compare against it before joining - this
        keeps casual joiners out, but anyone modifying their own client
        can skip it (the join itself can't be refused host-side)."""
        if not lobby_info["has_password"]:
            return True
        return cls.hash_password(lobby_info["pw_salt"], password) == lobby_info["pw_hash"]

    def host_lobby(self, name, map_key, max_players, password, on_result):
        """Creates a public lobby tagged with its settings. on_result(ok,
        info) is called once - info is the lobby id, or an error string."""
        if not self.client:
            on_result(False, "Steam isn't running")
            return
        self._pending_host = {
            "name": name, "map": map_key, "max_players": max_players,
            "password": password, "on_result": on_result,
        }
        self._schedule(0, lambda: self.client.create_lobby(2, max_players, self.on_lobby_created))

    def on_lobby_created(self, lobby_id, error=None):
        pending, self._pending_host = self._pending_host, None
        on_result = pending["on_result"] if pending else None
        if error or lobby_id is None:
            print(f"\n--> Failed to create lobby: {error}")
            if on_result:
                on_result(False, str(error))
            return
        self.current_lobby_id = lobby_id
        print(f"\n--> SUCCESS! Lobby Created ID: {self.current_lobby_id}")

        if pending:
            # GameIdentity is what py_steam_net's own get_lobby_list
            # filters by server-side - without it this lobby would be
            # invisible to every search, including our own.
            data = {
                "GameIdentity": "Rat King",
                "lobby_name": pending["name"],
                "map": pending["map"],
                "max_players": str(pending["max_players"]),
                "has_password": "1" if pending["password"] else "0",
            }
            if pending["password"]:
                salt = secrets.token_hex(8)
                data["pw_salt"] = salt
                data["pw_hash"] = self.hash_password(salt, pending["password"])
            for key, value in data.items():
                try:
                    self.client.set_lobby_data(lobby_id, key, value)
                except Exception as e:
                    print(f"Failed to set lobby data {key!r}: {e}")

        print("Hosting game session...")
        self._schedule(0.5, self.print_session_roster)
        if on_result:
            on_result(True, lobby_id)

    def request_lobbies(self, on_result):
        """Searches for open lobbies. on_result(lobbies, error): lobbies
        is a list of dicts (id, name, map, players, max_players,
        has_password, pw_salt, pw_hash), error is None or a string."""
        if not self.client:
            on_result([], "Steam isn't running")
            return

        def handle(lobby_ids, error):
            if error:
                on_result([], str(error))
            else:
                on_result([self._lobby_info(lobby_id) for lobby_id in lobby_ids], None)

        self._schedule(0, lambda: self.client.get_lobby_list(handle))

    def _lobby_info(self, lobby_id):
        def get(key):
            return self.client.get_lobby_data(lobby_id, key) or ""

        try:
            players = len(self.client.get_lobby_members(lobby_id))
        except Exception:
            players = 0
        try:
            max_players = int(get("max_players"))
        except ValueError:
            max_players = 0
        return {
            "id": lobby_id,
            "name": get("lobby_name") or "Unnamed lobby",
            "map": get("map"),
            "players": players,
            "max_players": max_players,
            "has_password": get("has_password") == "1",
            "pw_salt": get("pw_salt"),
            "pw_hash": get("pw_hash"),
        }

    def join_lobby(self, lobby_id, on_result):
        """on_result(ok, info) is called once - info is the lobby id, or
        an error string."""
        if not self.client:
            on_result(False, "Steam isn't running")
            return
        self._pending_join = on_result
        self._schedule(0, lambda: self.client.join_lobby(lobby_id, self.on_lobby_joined))

    def on_lobby_joined(self, lobby_id, error=None):
        on_result, self._pending_join = self._pending_join, None
        if error or lobby_id is None:
            print(f"\n--> Failed to join lobby: {error}")
            if on_result:
                on_result(False, str(error))
            return
        self.current_lobby_id = lobby_id
        print(f"\n--> SUCCESS! Joined Lobby ID: {self.current_lobby_id}")
        self._schedule(0.5, self.print_session_roster)
        if on_result:
            on_result(True, lobby_id)


    def on_lobby_changed(self, lobby_id, user_changed, making_change, member_state_change):
        print(f"\n[Lobby Update] Lobby ID: {lobby_id}, User: {user_changed}, State Change: {member_state_change}")
        if self.current_lobby_id:
            self._schedule(0.3, self.print_session_roster)

    # How many times setup_networking_mode retries an empty/failed
    # search before giving up and hosting - see that method's own
    # comment on why a single one-shot search isn't reliable. Spaced
    # _LOBBY_SEARCH_RETRY_DELAY_SECONDS apart, so worst case this adds
    # (_LOBBY_SEARCH_MAX_ATTEMPTS - 1) * _LOBBY_SEARCH_RETRY_DELAY_
    # SECONDS to how long a client with genuinely no one else to find
    # waits before hosting on its own.
    _LOBBY_SEARCH_MAX_ATTEMPTS = 5
    _LOBBY_SEARCH_RETRY_DELAY_SECONDS = 2.0

    # =============================================================
    # TRANSFORM SYNC
    # =============================================================

    def notify_shot(self, end_point=None):
        """Call when the local player fires. Sent as a running counter in the
        movement packets (so a lost packet can't drop a shot) - other players
        play the gunshot at our position when it goes up. end_point: where the
        bullet ended (its hit point, or the end of its range) - the last few
        travel in the packets too, so other players can draw the tracer."""
        self._shot_count += 1
        if end_point is not None:
            self._recent_shot_ends.append([round(end_point.x, 1), round(end_point.y, 1), round(end_point.z, 1)])
            del self._recent_shot_ends[:-3]

    def notify_footstep(self, material, volume):
        """Call when the local player's own footstep sounds (see app.py's own
        CharacterController.pop_footstep call) - sent as a counter, like jumps/shots, so a
        lost packet can't drop one; the (material, volume) of the LATEST footstep travels
        alongside it, since unlike a shot's end points a footstep has no further-back history
        worth keeping (RemotePlayer.receive_state only ever needs to play the newest one)."""
        self._footstep_count += 1
        self._last_footstep = (material or "", round(float(volume), 2))

    # How long after their last hit on us a shot still counts as the killing
    # blow, seconds - long enough that a death from bleed-out/fall/a laggy
    # last packet still credits the right person, short enough that an old,
    # unrelated hit from minutes ago can't retroactively "steal" a kill.
    KILL_CREDIT_WINDOW = 8.0

    def notify_death(self, push=None, damage_class="bullet"):
        """Call when the local player dies (their state packets then carry it: the
        counter other players play the gibs from, and the alive flag they hide the body by).
        Also broadcasts the kill (who did it, with what, whether it was a
        headshot) to the WHOLE lobby - see _broadcast_kill - if whoever hit us
        most recently did so recently enough (see KILL_CREDIT_WINDOW). A
        self-inflicted or environmental death (no recent attacker) broadcasts
        nothing - there's no kill to credit or show.

        damage_class: what killed us, by name (see Modules/Weapons/damage_classes.py's own
        DamageClass.name) - decides gibs vs. a dissolve on every client watching us die.
        Carried the same way push already is (see _death_push), read by RemotePlayer at the
        exact moment it sees this death, not before or after."""
        self._death_count += 1
        self.local_alive = False
        self._death_damage_class = str(damage_class or "bullet")
        # Which way an explosion threw us, so everyone's copy of our gibs flies the same way.
        self._death_push = [round(float(c), 2) for c in push] if push is not None else None
        attacker_id, attacker_time = self._last_attacker_id, self._last_attacker_time
        self._last_attacker_id = None
        if attacker_id is not None and time.perf_counter() - attacker_time <= self.KILL_CREDIT_WINDOW:
            self._broadcast_kill(attacker_id, self._last_attacker_weapon, self._last_attacker_headshot)

    def _broadcast_kill(self, killer_id, weapon_name, headshot):
        """Reliable 1-message announcement, to EVERY lobby member (not just
        killer_id), that WE just died to killer_id - the ONLY way anyone
        learns of this kill: each client is the sole authority on its own
        death, the same way it's already the sole authority on its own
        health/damage (see send_damage's own docstring). killer_id's own
        client is the one that turns this into its kill COUNT going up (see
        _receive_killfeed) - everyone else just shows it in their feed."""
        if not self.current_lobby_id:
            return
        payload = json.dumps(
            {"kf": 1, "k": int(killer_id), "w": str(weapon_name), "hs": int(bool(headshot))},
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            for member_id in self.client.get_lobby_members(self.current_lobby_id):
                if member_id != self.local_steam_id:
                    self.client.send_message_to(member_id, self.HELLO_FLAGS, 0, payload)
        except Exception:
            pass

    def notify_respawn(self):
        self.local_alive = True
        self._death_push = None

    def notify_emote(self, emote_name):
        """Reliable 1-message broadcast, to every OTHER lobby member, that we
        just started playing `emote_name` - same shape as _broadcast_kill
        (own-authority, discrete, reliable event; no counter/ack needed since
        there's nothing to lose that matters - a dropped emote packet just
        means that one peer doesn't see it play, same failure mode kill feed
        already accepts). "en" (not bare "n") for the name field - the regular
        movement packet already uses "n" for the sender's display name (see
        RemotePlayer.receive_state); this is a separate packet type handled by
        its own early-return branch in handle_data, so there's no actual
        collision, but a distinct key avoids confusing the two while reading
        the dispatch code."""
        if not self.current_lobby_id:
            return
        payload = json.dumps(
            {"em": 1, "en": str(emote_name)}, separators=(",", ":"),
        ).encode("utf-8")
        try:
            for member_id in self.client.get_lobby_members(self.current_lobby_id):
                if member_id != self.local_steam_id:
                    self.client.send_message_to(member_id, self.HELLO_FLAGS, 0, payload)
        except Exception:
            pass

    def send_damage(self, victim_id, amount, weapon_name="", headshot=False, push=None, damage_class="bullet",
                     origin=None):
        """Tells `victim_id` they were shot for `amount`: the shooter decides
        a hit (against the victim's hitbox as the shooter sees it) and the
        victim applies it to their own health - see on_damage. RELIABLE, unlike
        the movement stream: a hit must not be lost or reordered. headshot is
        carried along purely so the VICTIM can pass it back in the kill feed
        broadcast if this turns out to be the killing blow (see notify_death) -
        it has no effect on the damage itself (weapon.fire already folded any
        headshot multiplier into `amount` before this was even called).

        damage_class: the WEAPON's own damage_class.name (see Modules/Weapons/
        damage_classes.py) - sent explicitly rather than derived from weapon_name on the
        receiving end, since that's a cosmetic display name ("Gouda Gun"), not a registry id
        the weapon class could be looked back up from.

        origin: optional world-space (x, y, z) where the damage actually originated - the
        SHOOTER's own position at the moment of a direct hitscan hit (not the point the
        bullet struck the victim - a spot on their own hitbox a few tenths of a metre
        across that says nothing about the shooter's bearing from them, and only gets
        noisier, as a swing in indicated direction, the farther apart the two of them are),
        or an explosion's blast centre. Lets the victim's own damage-direction indicator
        (app.py's on_damaged/DamageIndicator) point at exactly where the shot/blast came
        from, rather than guessing from the attacker's CURRENT (replicated, possibly
        slightly stale) position, which is only a fallback for an older/odd message with no
        origin and can be badly wrong for splash damage - an explosion can reach someone
        standing well off to the side of wherever the attacker themselves is, who would
        otherwise see an arrow pointing at the attacker instead of the actual blast.

        Returns whether it was sent."""
        if victim_id == self.local_steam_id or not self.current_lobby_id:
            return False
        payload_dict = {"dmg": round(float(amount), 2), "w": str(weapon_name), "hs": int(bool(headshot)),
                        "dc": str(damage_class)}
        if push is not None:
            payload_dict["p"] = [round(float(c), 2) for c in push]   # an explosion's shove, for the victim's gibs
        if origin is not None:
            payload_dict["o"] = [round(float(c), 2) for c in origin]
        payload = json.dumps(
            payload_dict, separators=(",", ":"),
        ).encode("utf-8")
        try:
            self.client.send_message_to(victim_id, self.HELLO_FLAGS, 0, payload)
            return True
        except Exception:
            return False

    def _receive_damage(self, sender_id, message):
        try:
            amount = float(message["dmg"])
        except (KeyError, TypeError, ValueError):
            return
        amount = max(0.0, min(amount, 1000.0))   # it comes off the wire: keep it sane
        if amount > 0.0:
            self._last_attacker_id = sender_id
            self._last_attacker_time = time.perf_counter()
            self._last_attacker_weapon = str(message.get("w", ""))
            self._last_attacker_headshot = bool(message.get("hs", False))
            push = message.get("p")
            try:
                push = [max(-50.0, min(50.0, float(c))) for c in push][:3] if push is not None else None
            except (TypeError, ValueError):
                push = None
            if push is not None and len(push) != 3:
                push = None
            origin = message.get("o")
            try:
                origin = [float(c) for c in origin][:3] if origin is not None else None
            except (TypeError, ValueError):
                origin = None
            if origin is not None and len(origin) != 3:
                origin = None
            damage_class = str(message.get("dc", "bullet"))
            if self.on_damage is not None:
                self.on_damage(amount, sender_id, self._last_attacker_weapon, self._last_attacker_headshot,
                               push, damage_class, origin)

    def _receive_killfeed(self, sender_id, message):
        """sender_id is the VICTIM (see _broadcast_kill's own docstring -
        they're the one who sent this)."""
        try:
            killer_id = int(message.get("k", 0))
        except (TypeError, ValueError):
            return
        weapon_name = str(message.get("w", ""))
        headshot = bool(message.get("hs", False))
        if killer_id == self.local_steam_id:
            self._kill_count += 1
        if self.on_killfeed is not None:
            self.on_killfeed(killer_id, sender_id, weapon_name, headshot)

    def _receive_emote(self, sender_id, state):
        """Dispatches straight into the specific RemotePlayer instance (unlike
        on_killfeed/on_chat above, which are NetworkManager-level callbacks for
        a global UI event) - an emote is a per-player effect on that one
        player's own model, not something app.py-level UI code needs to react
        to generically."""
        rp = self.remote_players.get(sender_id)
        if rp is None:
            return   # a packet from a peer we haven't discovered via a movement packet yet
        name = str(state.get("en", ""))
        if name:
            rp.play_emote(name)

    def send_chat(self, text):
        """Broadcasts a chat message to every OTHER lobby member (not
        ourselves - the sender shows their own message locally right away
        instead, see ChatBox._submit, so there's no need to also receive it
        back over the wire). RELIABLE, like every other non-movement message
        here - a chat line must not be lost or reordered."""
        text = str(text)[:240]
        if not text or not self.current_lobby_id:
            return
        payload = json.dumps({"chat": 1, "m": text}, separators=(",", ":")).encode("utf-8")
        try:
            for member_id in self.client.get_lobby_members(self.current_lobby_id):
                if member_id != self.local_steam_id:
                    self.client.send_message_to(member_id, self.HELLO_FLAGS, 0, payload)
        except Exception:
            pass

    def send_admin_kill(self, victim_id, damage_class="bullet"):
        """The /kill (and /smite) command's remote-target path (see Modules/Chat/
        commands.py and app.py's own kill_player/smite_player): tells victim_id an admin
        wants them dead. RELIABLE. The RECEIVING client is what actually
        decides to die (same self-authority principle send_damage's own
        docstring already documents) - and specifically only if the sender
        really is commands.ADMIN_STEAM_ID (see _receive_admin_kill), which
        every client can check for itself against that same hardcoded
        constant, so a non-admin spoofing this message accomplishes nothing.

        damage_class: whether to die smite-self's way (Zap - the lightning effect and a
        dissolve instead of gibs, see app.py's own smite_self) or a plain kill (Bullet,
        the default) - carried in the same message rather than a separate one, since it's
        still exactly one "you should die right now, this way" request either way."""
        if victim_id == self.local_steam_id or not self.current_lobby_id:
            return False
        payload = json.dumps({"ak": 1, "dc": str(damage_class)}, separators=(",", ":")).encode("utf-8")
        try:
            self.client.send_message_to(victim_id, self.HELLO_FLAGS, 0, payload)
            return True
        except Exception:
            return False

    def _receive_admin_kill(self, sender_id, damage_class="bullet"):
        if self.on_admin_kill is not None:
            self.on_admin_kill(sender_id, damage_class)

    def _receive_chat(self, sender_id, message):
        text = str(message.get("m", ""))[:240]
        if text and self.on_chat is not None:
            self.on_chat(sender_id, text)

    def display_name(self, steam_id):
        """The name to show for steam_id anywhere in the UI (scoreboard, kill
        feed, name tags) - "You"/local_name for ourselves, a remote player's
        own replicated name, or a placeholder if neither is known yet."""
        if steam_id == self.local_steam_id:
            return self.local_name or "You"
        player = self.remote_players.get(steam_id)
        if player is not None and player.name:
            return player.name
        return f"Player {steam_id % 10000}"

    @property
    def local_kills(self):
        """Kills the local player has been credited for - see notify_death/
        _broadcast_kill/_receive_killfeed. Scoreboard/GameMode.standings()
        read this rather than the private counter directly."""
        return self._kill_count

    @property
    def local_deaths(self):
        """Times the local player has died - see notify_death."""
        return self._death_count

    def set_local_state(self, feet_pos, yaw, pitch, speed, crouched, grounded,
                        sprinting, move_direction, jumped):
        """Call once per frame with the LOCAL player's real state (not the
        camera - in third person the camera sits on a boom arm behind the
        player, so broadcasting it would put everyone else's view of you
        in the wrong place). This is exactly what drives the local
        PlayerModel, so remote players' copies of you animate the same
        way. jumped: True on the frame a jump actually executed - it's
        sent as a counter (see RemotePlayer.receive_state) so a lost
        packet can't drop the event."""
        if jumped:
            self._jump_count += 1
        self._local_state = {
            "p": [round(feet_pos.x, 3), round(feet_pos.y, 3), round(feet_pos.z, 3)],
            "y": round(yaw, 2),
            "pt": round(pitch, 2),
            "v": round(speed, 2),
            "c": int(crouched),
            "g": int(grounded),
            "s": int(sprinting),
            "d": [round(move_direction.x, 2), round(move_direction.z, 2)],
            "j": self._jump_count,
            "h": self.local_hat or "",
            "wp": self.local_weapon or "",
            "k": encode_color(self.local_color),
            "f": self._shot_count,
            "e": self._recent_shot_ends,
            "fs": self._footstep_count,
            "fm": self._last_footstep[0],
            "fv": self._last_footstep[1],
            "x": self._death_count,
            "dc": self._death_damage_class,
            "ki": self._kill_count,
            "a": int(self.local_alive),
            "n": self.local_name,
        }
        if self._death_push is not None:
            self._local_state["b"] = self._death_push

    HELLO_INTERVAL = 0.5
    HELLO_FLAGS = 8 | 32  # Reliable | AutoRestartBrokenSession

    def _initiates_to(self, member_id):
        """Exactly ONE side of each pair may open the Steam session (both
        sending first = crossed connection attempts Steam can't resolve), and
        it must be the side that is ready to answer, not the one still
        loading: a joiner opens the session to the host (first lobby member);
        the host stays silent toward a peer until it has heard from them.
        Two non-hosts: the lower Steam id opens it."""
        if member_id in self.remote_players or member_id in self._heard_from:
            return True
        host_id = self._member_order[0] if self._member_order else None
        if member_id == host_id:
            return True
        if self.local_steam_id == host_id:
            return False
        return self.local_steam_id < member_id

    def _friend_name(self, steam_id):
        """Steam's own record of a player's name ('' if not loaded yet) - only
        a fallback until their packets, which carry the name, arrive."""
        try:
            return self.client.friend_name(steam_id)
        except Exception:
            return ""

    def _refresh_names(self):
        """Steam loads persona names lazily (non-friends most of all), so a
        name that was empty when first asked for is asked for again here -
        ours (sent in every packet) and any remote player's still-blank one."""
        if not self.local_name:
            try:
                self.local_name = self.client.own_name()
            except Exception:
                pass
            if not self.local_name:
                self.local_name = self._friend_name(self.local_steam_id)
            if self.local_name:
                print(f"Your Steam name: {self.local_name}")
        for steam_id, remote in self.remote_players.items():
            if not remote.name:
                remote.name = self._friend_name(steam_id)

    def avatar_rgba(self, steam_id):
        """64x64 RGBA bytes of a player's Steam profile picture, or None while
        Steam is still fetching it (or the py_steam_net build predates
        friend_avatar). Cached once found; misses retry at most once a second."""
        if steam_id in self._avatars:
            return self._avatars[steam_id]
        now = time.perf_counter()
        if now - self._avatar_tried.get(steam_id, -1.0) < 1.0:
            return None
        self._avatar_tried[steam_id] = now
        try:
            data = self.client.friend_avatar(steam_id)
        except Exception:
            return None
        if data:
            self._avatars[steam_id] = data
        return data or None

    def _relay_ready(self):
        """Steam's relay network takes several seconds after launch to become
        usable ("Current"); connecting before that fails with ConnectFailed.
        py_steam_net starts it at init, so by the time a lobby is joined this
        is normally already true. Older wheels without relay_status: assume yes."""
        if self._relay_ok:
            return True
        try:
            status = self.client.relay_status()
        except Exception:
            self._relay_ok = True
            return True
        self._relay_ok = "Current" in status
        return self._relay_ok

    def _send_hellos(self, now):
        """Opens (and keeps retrying) a Steam session with every lobby member
        we haven't heard a movement packet from yet. The movement stream is
        UNRELIABLE, and Steam drops unreliable messages while a session is
        still being negotiated (relay handshake - seconds, not ms), which is
        why players used to take ~10s to appear on each other's screens. A
        RELIABLE message is queued until the session is up, then delivered,
        so it both forces the handshake immediately (even while the map is
        still loading) and can't be lost. Stops per-peer as soon as they show
        up in remote_players."""
        if not self.in_game or now - self._last_hello_time < self.HELLO_INTERVAL:
            return  # not in_game = still loading, can't answer a session request yet
        if not self._relay_ready():
            return  # Steam's relay network isn't up yet - a send now just fails
        self._last_hello_time = now
        members = self._refresh_members()
        if not members:
            return
        state = self._local_state if self.in_game and self._local_state is not None else {"hello": 1}
        payload = json.dumps(state, separators=(",", ":")).encode("utf-8")
        for member_id in members:
            if member_id == self.local_steam_id or member_id in self.remote_players:
                continue
            if not self._initiates_to(member_id):
                continue
            if now < self._retry_at.get(member_id, 0.0):
                continue
            if member_id not in self._net_log_t0:
                self._net_log_t0[member_id] = now
                status = self.client.relay_status() if hasattr(self.client, "relay_status") else "unknown"
                print(f"[Net] hello -> {member_id} (relay network: {status})")
            try:
                self.client.send_message_to(member_id, self.HELLO_FLAGS, 0, payload)
            except Exception:
                pass

    def _broadcast_transform(self, now):
        if not self.current_lobby_id or self._local_state is None:
            return
        # Rate-limited: the game loop runs at hundreds of fps but ~30
        # updates/s is plenty (receivers interpolate) - unthrottled, this
        # queued a packet per member per FRAME. Sent when the state changed
        # or as a heartbeat, so a standing-still player still reads as
        # connected and its last state is refreshed if a packet is lost.
        interval = now - self._last_send_time
        if interval < self.SEND_INTERVAL:
            return
        if self._local_state == self._last_sent_state and interval < self.HEARTBEAT_INTERVAL:
            return

        payload = json.dumps(self._local_state, separators=(",", ":")).encode("utf-8")
        try:
            for member_id in self.client.get_lobby_members(self.current_lobby_id):
                if (member_id != self.local_steam_id and
                        (member_id in self.remote_players or member_id in self._heard_from)):
                    # 1 = Steam's UnreliableNoNagle: movement is a stream
                    # where only the newest packet matters, so it must not
                    # queue/retransmit. (This used to send flag 2, which
                    # isn't a valid Steam flag - the binding fell back to
                    # RELIABLE, so every position update waited behind the
                    # last one.)
                    self.client.send_message_to(member_id, 1, 0, payload)
                    if self.profiler is not None:
                        self.profiler.count("packets out")
                        self.profiler.count("bytes out", len(payload))
        except Exception:
            pass
        self._last_sent_state = self._local_state
        self._last_send_time = now

    def _prune_remote_players(self):
        """Removes the model/hitbox of any peer no longer in the lobby."""
        members = self._refresh_members()
        if members is None:
            return
        for steam_id in list(self.remote_players):
            if steam_id not in members:
                print(f"\n--> Peer left the lobby: {steam_id}")
                self.remote_players.pop(steam_id).destroy()

    def _refresh_members(self):
        """Re-reads the current lobby's member ids. Returns the set, or None
        if it couldn't be read (left as-is in that case)."""
        try:
            order = list(self.client.get_lobby_members(self.current_lobby_id))
            self._member_order = order
            self._members = set(order)
        except Exception:
            return None
        self._members_checked = time.perf_counter()
        return self._members

    def _is_lobby_member(self, steam_id):
        """py_steam_net accepts every incoming Steam session (see its
        session_request_callback), and App 480 is shared by countless
        unrelated test projects - so only trust packets from people who are
        actually in OUR lobby. Refreshed at most once a second, so a stranger
        spamming packets can't make this a per-packet lobby query."""
        if steam_id in self._members:
            return True
        if time.perf_counter() - self._members_checked >= 1.0:
            self._refresh_members()
        return steam_id in self._members

    def handle_data(self, sender_id, ch, msg_bytes):
        if self.profiler is not None:
            self.profiler.count("packets in")
            self.profiler.count("bytes in", len(msg_bytes))
        if sender_id not in self._net_seen:
            self._net_seen.add(sender_id)
            t0 = self._net_log_t0.get(sender_id)
            since = f"{time.perf_counter() - t0:.1f}s after our first hello" if t0 else "before we sent any hello"
            print(f"[Net] first packet from {sender_id}: {since} (scene_ready={self.scene is not None}, member={sender_id in self._members})")
        if self.scene is None:
            return  # a peer's packet arrived before our map finished loading
        if not self._is_lobby_member(sender_id):
            return
        self._heard_from.add(sender_id)
        try:
            state = json.loads(msg_bytes.decode("utf-8"))
            if "dmg" in state:
                self._receive_damage(sender_id, state)
                return
            if "kf" in state:
                self._receive_killfeed(sender_id, state)
                return
            if "chat" in state:
                self._receive_chat(sender_id, state)
                return
            if "ak" in state:
                self._receive_admin_kill(sender_id, str(state.get("dc", "bullet")))
                return
            if "em" in state:
                self._receive_emote(sender_id, state)
                return
            if "p" not in state or "y" not in state:
                return  # not this version's movement packet
            if sender_id not in self.remote_players:
                print(f"\n--> Discovered peer in lobby: {sender_id}")
                self.remote_players[sender_id] = RemotePlayer(self.scene, sender_id)
                self.remote_players[sender_id].name = self._friend_name(sender_id)
                self.remote_players[sender_id].on_death = (
                    lambda *args: self.on_death(*args) if self.on_death else None)
                self.remote_players[sender_id].on_tracer = (
                    lambda *args: self.on_tracer(*args) if self.on_tracer else None)
                self.remote_players[sender_id].on_footstep = (
                    lambda *args: self.on_footstep(*args) if self.on_footstep else None)
                self.remote_players[sender_id].on_dissolve_spark = (
                    lambda *args: self.on_dissolve_spark(*args) if self.on_dissolve_spark else None)
                self.remote_players[sender_id].on_dissolve_arc = (
                    lambda *args: self.on_dissolve_arc(*args) if self.on_dissolve_arc else None)
            self.remote_players[sender_id].receive_state(state)
        except Exception as e:
            print(f"Error parsing incoming packet from {sender_id}: {e}")
