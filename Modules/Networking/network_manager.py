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

        try:
            self.client = py_steam_net.PySteamClient()
            self.client.init(480)
            print(f"Steam P2P initialized successfully. Ready: {self.client.is_ready()}")

            self.local_steam_id = self.client.own_steam_id()
            print(f"Your Steam ID: {self.local_steam_id}")

            self.client.set_message_recv_callback(self.handle_data)
            self.client.set_lobby_changed_callback(self.on_lobby_changed)
            self.client.set_connection_failed_callback(self.on_session_failed)
        except Exception as e:
            print(f"Steam initialization failed: {e}. Make sure Steam client is running.")
            self.client = None

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

    def update(self):
        if not self.client:
            return
        self.client.run_callbacks()
        self._run_deferred()
        if self.current_lobby_id:
            self._send_hellos(time.perf_counter())
        if self.in_game:
            # handle_data only ever runs from here: py_steam_net hands over
            # incoming packets when (and only when) they're polled for. Nothing
            # polled before this - other players could join the lobby but their
            # positions were never read, so their models never appeared.
            self.client.receive_messages(0, self.RECEIVE_BATCH)
            now = time.perf_counter()
            dt = now - self._last_update_time if self._last_update_time else 0.0
            self._last_update_time = now
            self._broadcast_transform(now)
            for remote in self.remote_players.values():
                remote.update(dt)
            if now - self._last_prune_time >= 1.0:
                self._last_prune_time = now
                self._prune_remote_players()

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
        }

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
            if "p" not in state or "y" not in state:
                return  # not this version's movement packet
            if sender_id not in self.remote_players:
                print(f"\n--> Discovered peer in lobby: {sender_id}")
                self.remote_players[sender_id] = RemotePlayer(self.scene, sender_id)
            self.remote_players[sender_id].receive_state(state)
        except Exception as e:
            print(f"Error parsing incoming packet from {sender_id}: {e}")
