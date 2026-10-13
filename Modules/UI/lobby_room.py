"""
The fullscreen waiting room shown between "a lobby exists" and "the match
actually starts" - see LobbyMenu's own docstring for exactly where this
fits (on_lobby_ready, not on_start, is what shows this now). Settings are
visible to everyone, editable only by the host (live, via NetworkManager.
update_lobby_setting - not just the one-time write host_lobby's own
on_lobby_created does); the player list (name + Steam avatar, same
pattern scoreboard.py already uses) updates on a short poll while
waiting. The host's Start button writes NetworkManager.start_the_game's
own flag - every peer (including the host) polls for it and, once seen,
hands off to on_game_start (the real on_start app.py already had before
this screen existed).

Deliberately a solid full-screen takeover (not a translucent overlay over
the 3D rat preview like LobbyMenu's own sidebar) - a waiting room reads
better as its own dedicated screen than as another card floating over the
main menu's background.
"""

from . import theme
from .controls import ScrollBox
from .inputs import Dropdown, TextInput
from .widgets import Anchor, Button, Image, Label, Panel
from .scoreboard import _fit_font_size, _text_width

_CARD_W = 520
AVATAR_SIZE = 56
AVATAR_PAD = 8
AVATAR_PIXELS = (64, 64)   # what NetworkManager.avatar_rgba hands back
NAME_COLUMN_WIDTH = 260
TAG_WIDTH = 70   # reserved out of NAME_COLUMN_WIDTH for "(you)"/"(host)"


def _button(text, on_click, **kw):
    return Button(text, on_click=on_click, **theme.BUTTON, **kw)


def _index_for_key(pairs, key):
    for i, (_name, k) in enumerate(pairs):
        if k == key:
            return i
    return 0


class LobbyRoom:
    POLL_SECONDS = 1.0
    MAX_PLAYERS = 10

    def __init__(self, ui, net, maps, modes, on_game_start, on_leave):
        """maps/modes: same [(display name, key), ...] shape LobbyMenu
        already takes. on_game_start(map_key, mode_key): fires once the
        host presses Start (seen via NetworkManager.is_game_started) -
        this is where the caller actually transitions into gameplay
        (app.py's own request_start). on_leave(): fires when Back/Leave
        is pressed, or set_visible(False) is otherwise appropriate -
        the caller shows the main menu again."""
        self.ui = ui
        self.net = net
        self.maps = maps
        self.modes = modes
        self._map_names = {key: name for name, key in maps}
        self._mode_names = {key: name for name, key in modes}
        self.on_game_start = on_game_start
        self.on_leave = on_leave
        self._since_poll = 0.0
        self._is_host_built_for = None   # which role the settings card was last built for
        self._player_signature = None
        self._game_started_handled = False
        self.root = self._build()

    # ---- construction ---------------------------------------------------

    def _build(self):
        # theme.SIDEBAR, not a solid colour: it is translucent like the join panel, so the menu scene behind
        # (the skybox and the preview rat, still being rendered while the lobby is up) shows through.
        root = Panel(color=theme.SIDEBAR, size_frac=(1, 1), visible=False, name="lobby_room")
        root.add(Label("LOBBY", font_size=64, color=theme.ACCENT, font=theme.title_font_path(),
                       anchor=Anchor.TOP_CENTER, offset=(0, 36)))

        body = root.add(Panel(layout="horizontal", spacing=40, fit_content=True,
                              anchor=Anchor.CENTER, offset=(0, 10)))
        self.settings_card = body.add(Panel(color=theme.CARD, layout="vertical", spacing=12,
                                            padding=22, fit_content=True, size=(_CARD_W, 0)))
        self.players_card = body.add(Panel(color=theme.CARD, layout="vertical", spacing=12,
                                           padding=22, fit_content=True, size=(_CARD_W, 0)))
        self.players_card.add(Label("PLAYERS", font_size=26, color=theme.ACCENT))
        self.player_list = self.players_card.add(ScrollBox(size=(_CARD_W - 44, 380), padding=6, spacing=8))

        footer = root.add(Panel(layout="horizontal", spacing=20, fit_content=True,
                                anchor=Anchor.BOTTOM_CENTER, offset=(0, -40)))
        footer.add(_button("LEAVE", self._press_leave, size=(220, 70), font_size=32))
        self.start_button = footer.add(_button("START", self._press_start, size=(220, 70), font_size=32))
        return root

    def _build_settings_card(self, is_host):
        """Rebuilt whenever is_host actually changes (e.g. the original
        host disconnects mid-wait and someone else's is_host flips True -
        see update() below) - host gets live-editable controls, everyone
        else gets plain read-only labels for the same values. Widgets
        have no built-in disabled mode, so two different child sets per
        role is the simpler/lower-risk choice over adding one."""
        self._is_host_built_for = is_host
        self.settings_card.clear_children()
        card = self.settings_card
        card.add(Label("SETTINGS", font_size=26, color=theme.ACCENT))
        info = self.net.current_lobby_info() or {}

        if is_host:
            card.add(Label("Lobby name", font_size=20, color=theme.MUTED))
            self.name_input = card.add(TextInput(
                info.get("name", "Rat King Lobby"), max_length=24, size=(_CARD_W - 44, 50), font_size=24,
                on_submit=lambda text: self.net.update_lobby_setting("lobby_name", text or "Rat King Lobby"),
            ))
            card.add(Label("Map", font_size=20, color=theme.MUTED))
            self.map_dropdown = card.add(Dropdown(
                [name for name, _ in self.maps], selected=_index_for_key(self.maps, info.get("map")),
                on_change=lambda i, _o: self.net.update_lobby_setting("map", self.maps[i][1]),
                size=(_CARD_W - 44, 50), font_size=24,
            ))
            card.add(Label("Game mode", font_size=20, color=theme.MUTED))
            self.mode_dropdown = card.add(Dropdown(
                [name for name, _ in self.modes], selected=_index_for_key(self.modes, info.get("mode")),
                on_change=lambda i, _o: self.net.update_lobby_setting("mode", self.modes[i][1]),
                size=(_CARD_W - 44, 50), font_size=24,
            ))
            if info.get("has_password"):
                card.add(Label("Password protected (set at creation)", font_size=18, color=theme.MUTED))
        else:
            self.name_label = card.add(Label("", font_size=24, color=theme.TEXT))
            self.map_label = card.add(Label("", font_size=24, color=theme.TEXT))
            self.mode_label = card.add(Label("", font_size=24, color=theme.TEXT))
            self.players_count_label = card.add(Label("", font_size=24, color=theme.TEXT))
            self.password_label = card.add(Label("", font_size=20, color=theme.MUTED))
            self._apply_readonly_settings(info)

        self.start_button.visible = is_host

    def _apply_readonly_settings(self, info):
        self.name_label.text = info.get("name", "...")
        self.map_label.text = f"Map: {self._map_names.get(info.get('map'), info.get('map', '...'))}"
        self.mode_label.text = f"Mode: {self._mode_names.get(info.get('mode'), info.get('mode', '...'))}"
        self.players_count_label.text = f"Players: {info.get('players', 0)}/{info.get('max_players') or '?'}"
        self.password_label.text = "Password protected" if info.get("has_password") else ""

    # ---- player list ------------------------------------------------

    def _refresh_players(self):
        self.net.refresh_members()
        order = list(self.net._member_order) or [self.net.local_steam_id]
        rows = []
        for steam_id in order:
            if steam_id == self.net.local_steam_id:
                name = self.net.local_name or "..."
            else:
                name = self.net._friend_name(steam_id) or "..."
            rows.append((steam_id, name, self.net.avatar_rgba(steam_id)))

        host_id = self.net.host_steam_id
        signature = (host_id,) + tuple((sid, name, avatar is not None) for sid, name, avatar in rows)
        if signature == self._player_signature:
            return
        self._player_signature = signature
        self.player_list.clear_children()
        for steam_id, name, avatar in rows:
            self.player_list.add(self._player_row(steam_id, name, avatar, is_host=(steam_id == host_id)))

    def _player_row(self, steam_id, name, avatar, is_host):
        mine = steam_id == self.net.local_steam_id
        row = Panel(color=(34, 38, 56, 240) if mine else (24, 27, 42, 240), padding=8,
                   size=(_CARD_W - 44, AVATAR_SIZE + 2 * AVATAR_PAD))
        if avatar:
            pic = row.add(Image(size=(AVATAR_SIZE, AVATAR_SIZE), anchor=Anchor.MIDDLE_LEFT,
                                offset=(AVATAR_PAD, 0)))
            pic.set_rgba(AVATAR_PIXELS, avatar)
        else:
            row.add(Panel(color=theme.CARD_BORDER, size=(AVATAR_SIZE, AVATAR_SIZE),
                          anchor=Anchor.MIDDLE_LEFT, offset=(AVATAR_PAD, 0)))

        tag = " (you)" if mine else (" (host)" if is_host else "")
        name_x = AVATAR_PAD + AVATAR_SIZE + 14
        manager = self.players_card.manager
        budget = NAME_COLUMN_WIDTH - (TAG_WIDTH if tag else 0)
        name_size = _fit_font_size(manager, name, budget)
        row.add(Label(name, font_size=name_size, color=theme.TEXT, anchor=Anchor.MIDDLE_LEFT,
                      offset=(name_x, 0)))
        if tag:
            name_w = _text_width(manager, name, name_size)
            row.add(Label(tag, font_size=18, color=theme.MUTED, anchor=Anchor.MIDDLE_LEFT,
                          offset=(name_x + name_w + 4, 0)))
        return row

    # ---- lifecycle ----------------------------------------------------

    def show(self, map_key, mode_key):
        """Called once, right when hosting/joining succeeds (see
        LobbyMenu's own on_lobby_ready wiring) - map_key/mode_key here
        are only the STARTING values (what was just created/what the
        joined lobby already had); from this point on the host's own
        live edits (update_lobby_setting) are the source of truth, read
        back fresh on every poll via current_lobby_info()."""
        self.net.clear_game_started()
        self._game_started_handled = False
        self._since_poll = 0.0
        self._player_signature = None
        self._is_host_built_for = None
        self._build_settings_card(self.net.is_host)
        self._refresh_players()
        self.root.visible = True

    def hide(self):
        self.root.visible = False

    def _press_start(self):
        self.net.start_the_game()

    def _press_leave(self):
        self.net.leave_lobby()
        self.hide()
        self.on_leave()

    def update(self, dt):
        if not self.root.visible:
            return
        self._since_poll += dt
        if self._since_poll < self.POLL_SECONDS:
            return
        self._since_poll = 0.0

        if self.net.is_host != self._is_host_built_for:
            self._build_settings_card(self.net.is_host)
        if not self._is_host_built_for:
            self._apply_readonly_settings(self.net.current_lobby_info() or {})
        self._refresh_players()

        if not self._game_started_handled and self.net.is_game_started():
            self._game_started_handled = True
            info = self.net.current_lobby_info() or {}
            map_key = info.get("map") or next(iter(self._map_names), None)
            mode_key = info.get("mode") or "deathmatch"
            self.hide()
            self.on_game_start(map_key, mode_key)
