import json
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

    def __init__(self, camera, scene):
        self.camera = camera
        self.scene = scene
        self.remote_players = {}
        self.current_lobby_id = None
        self.last_pos = None
        self.last_hpr = None
        self._deferred = []  # list of (fire_time, func) - replaces taskMgr.doMethodLater

        try:
            self.client = py_steam_net.PySteamClient()
            self.client.init(480)
            print(f"Steam P2P initialized successfully. Ready: {self.client.is_ready()}")

            self.local_steam_id = self.client.own_steam_id()
            print(f"Your Steam ID: {self.local_steam_id}")

            self.client.set_message_recv_callback(self.handle_data)
            self.client.set_lobby_changed_callback(self.on_lobby_changed)

            self.setup_networking_mode()

        except Exception as e:
            print(f"Steam initialization failed: {e}. Make sure Steam client is running.")
            self.client = None

    # =============================================================
    # PER-FRAME UPDATE - call this once per frame from your main loop
    # =============================================================

    def update(self):
        if not self.client:
            return
        self.client.run_callbacks()
        self._run_deferred()
        self._broadcast_transform()

    def _schedule(self, delay_seconds, func):
        """Replaces taskMgr.doMethodLater - runs func() once, after at
        least delay_seconds have passed, the next time update() runs."""
        self._deferred.append((time.monotonic() + delay_seconds, func))

    def _run_deferred(self):
        if not self._deferred:
            return
        now = time.monotonic()
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

    def on_lobby_created(self, lobby_id, error=None):
        if error:
            print(f"\n--> Failed to create lobby: {error}")
        else:
            self.current_lobby_id = lobby_id
            print(f"\n--> SUCCESS! Lobby Created ID: {self.current_lobby_id}")

            # Tag the lobby specifically for your game so strangers' Spacewar lobbies are ignored
            try:
                self.client.set_lobby_data(lobby_id, "gname", "RatWarGame")
            except Exception as e:
                print(f"Failed to set custom lobby tag: {e}")

            print("Hosting game session...")
            self._schedule(0.5, self.print_session_roster)

    def on_lobby_joined(self, lobby_id, error=None):
        if error:
            print(f"\n--> Failed to join lobby: {error}")
        else:
            self.current_lobby_id = lobby_id
            print(f"\n--> SUCCESS! Joined Lobby ID: {self.current_lobby_id}")
            self._schedule(0.5, self.print_session_roster)

    def on_lobby_changed(self, lobby_id, user_changed, making_change, member_state_change):
        print(f"\n[Lobby Update] Lobby ID: {lobby_id}, User: {user_changed}, State Change: {member_state_change}")
        if self.current_lobby_id:
            self._schedule(0.3, self.print_session_roster)

    def setup_networking_mode(self):
        print("\n--- Scanning for Open RatWar Lobbies ---")

        def handle_lobby_list(lobbies, error):
            if error:
                print(f"Failed to request lobby list: {error}")
                print("Hosting a new lobby instead...")
                self.client.create_lobby(2, 4, self.on_lobby_created)
                return

            open_lobby_id = None
            if lobbies:
                for l_id in lobbies:
                    try:
                        members = self.client.get_lobby_members(l_id)
                        if members and 0 < len(members) < 4:
                            open_lobby_id = l_id
                            break
                    except Exception:
                        continue

            if open_lobby_id:
                print(f"Found valid open lobby {open_lobby_id}. Joining automatically...")
                self.client.join_lobby(open_lobby_id, self.on_lobby_joined)
            else:
                print("No open RatWar lobbies found. Hosting a new lobby...")
                self.client.create_lobby(2, 4, self.on_lobby_created)

        try:
            self.client.get_lobby_list(handle_lobby_list)
        except Exception as e:
            print(f"Error initiating lobby list request: {e}")
            self.client.create_lobby(2, 4, self.on_lobby_created)

    # =============================================================
    # TRANSFORM SYNC
    # =============================================================

    def _broadcast_transform(self):
        if not self.current_lobby_id:
            return

        current_pos = (self.camera.position.x, self.camera.position.y, self.camera.position.z)
        current_hpr = (self.camera.yaw, self.camera.pitch, 0.0)

        if current_pos != self.last_pos or current_hpr != self.last_hpr:
            payload = json.dumps({
                "pos": list(current_pos),
                "hpr": list(current_hpr)
            }).encode("utf-8")

            try:
                members = self.client.get_lobby_members(self.current_lobby_id)
                for member_id in members:
                    if member_id != self.local_steam_id:
                        self.client.send_message_to(member_id, 2, 0, payload)
            except Exception:
                pass

            self.last_pos = current_pos
            self.last_hpr = current_hpr

    def handle_data(self, sender_id, ch, msg_bytes):
        if sender_id not in self.remote_players:
            print(f"\n--> Discovered peer in lobby: {sender_id}")
            self.remote_players[sender_id] = RemotePlayer(self.scene, sender_id)

        try:
            parsed = json.loads(msg_bytes.decode("utf-8"))
            if "pos" in parsed and "hpr" in parsed:
                self.remote_players[sender_id].update_transform(
                    parsed["pos"], parsed["hpr"]
                )
        except Exception as e:
            print(f"Error parsing incoming packet from {sender_id}: {e}")