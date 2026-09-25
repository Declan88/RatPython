"""
The main menu's Host / Join flow.

Host opens a settings panel (lobby name, map dropdown, optional password,
player count 2-10); Create asks NetworkManager to host and then calls
on_start(map_key). Join opens a searchable, scrollable list of open lobbies
(auto-refreshing while open); picking one - after a password prompt if it
has one - joins it and calls on_start with that lobby's map.

NetworkManager's on_result callbacks can run inside Steam's callback
dispatch, so everything they reach here (_on_lobbies/_on_hosted/_on_joined)
only updates widgets/flags - on_start itself is expected to defer the real
work (app.py just records a pending start for its main loop).
"""

from .controls import ScrollBox, Slider
from .inputs import Dropdown, TextInput
from .widgets import Anchor, Button, Label, Panel

_W = 524  # inner width of the host/join panels (560 - 2 * 18 padding)
_MUTED = (160, 160, 172, 255)
_WARN = (255, 210, 120, 255)


class LobbyMenu:
    REFRESH_SECONDS = 6.0
    MIN_PLAYERS = 2
    MAX_PLAYERS = 10

    def __init__(self, ui, net, maps, on_start):
        """maps: [(display name, scene key), ...]; on_start(scene_key)."""
        self.ui = ui
        self.net = net
        self.maps = maps
        self._map_names = {key: name for name, key in maps}
        self.on_start = on_start
        self.lobbies = []
        self._searching = False
        self._busy = False
        self._since_refresh = 0.0
        self._selected = None  # listed lobby waiting on a password
        self.root = self._build()

    # ---- construction ---------------------------------------------------

    def _build(self):
        root = Panel(size_frac=(1, 1), name="main_menu")
        column = root.add(Panel(anchor=Anchor.TOP_CENTER, offset=(0, 90), layout="vertical",
                                spacing=16, align="center", fit_content=True))
        column.add(Label("RATWAR", font_size=96))
        buttons = column.add(Panel(layout="horizontal", spacing=20, fit_content=True))
        buttons.add(Button("Host", on_click=self.toggle_host, size=(250, 60), font_size=32))
        buttons.add(Button("Join", on_click=self.toggle_join, size=(250, 60), font_size=32))
        self.status = column.add(Label("", font_size=22, color=_WARN, align="center"))
        self.host_panel = column.add(self._build_host_panel())
        self.join_panel = column.add(self._build_join_panel())
        return root

    def _panel(self):
        return Panel(color=(15, 15, 25, 225), layout="vertical", spacing=10, padding=18,
                     fit_content=True, size=(_W + 36, 0), visible=False)

    def _build_host_panel(self):
        p = self._panel()
        p.add(Label("Lobby name", font_size=20, color=_MUTED))
        self.name_input = p.add(TextInput("RatWar Lobby", max_length=24, size=(_W, 40)))
        p.add(Label("Map", font_size=20, color=_MUTED))
        self.map_dropdown = p.add(Dropdown([name for name, _ in self.maps], size=(_W, 40)))
        p.add(Label("Password (optional)", font_size=20, color=_MUTED))
        self.pass_input = p.add(TextInput(password=True, placeholder="No password",
                                          max_length=24, size=(_W, 40)))
        self.players_label = p.add(Label("", font_size=20, color=_MUTED))
        self.players_slider = p.add(Slider(4, self.MIN_PLAYERS, self.MAX_PLAYERS, step=1,
                                           size=(_W, 28), on_change=self._on_players))
        self._on_players(4)
        p.add(Button("Create lobby", on_click=self._create, size=(_W, 52), font_size=28))
        return p

    def _build_join_panel(self):
        p = self._panel()
        search = p.add(Panel(layout="horizontal", spacing=10, fit_content=True))
        self.search_input = search.add(TextInput(placeholder="Search lobby name...",
                                                 max_length=24, size=(400, 40),
                                                 on_change=lambda _t: self._rebuild_rows()))
        search.add(Button("Refresh", on_click=self.refresh, size=(_W - 410, 40), font_size=22))

        # Header columns line up with _make_row's label anchors: the list's
        # rows sit 6px in (ScrollBox padding) and are 502 wide (524 minus
        # padding and the scrollbar).
        header = p.add(Panel(size=(502, 22), offset=(6, 0)))
        header.add(Label("Lobby", font_size=18, color=_MUTED, anchor=(0, 0.5), offset=(12, 0)))
        header.add(Label("Map", font_size=18, color=_MUTED, anchor=(0.56, 0.5), pivot=(0, 0.5)))
        header.add(Label("Players", font_size=18, color=_MUTED, anchor=(0.86, 0.5), pivot=(0, 0.5)))

        self.list = p.add(ScrollBox(size=(_W, 260), padding=6, spacing=6))
        self.list_status = p.add(Label("", font_size=20, color=_MUTED))

        self.password_panel = p.add(Panel(layout="vertical", spacing=8, fit_content=True,
                                          visible=False))
        self.password_prompt = self.password_panel.add(Label("", font_size=20, color=_WARN))
        entry = self.password_panel.add(Panel(layout="horizontal", spacing=10, fit_content=True))
        self.join_pass_input = entry.add(TextInput(password=True, placeholder="Password",
                                                   max_length=24, size=(320, 40),
                                                   on_submit=lambda _t: self._submit_password()))
        entry.add(Button("Join", on_click=self._submit_password, size=(96, 40), font_size=22))
        entry.add(Button("Cancel", on_click=self._cancel_password, size=(88, 40), font_size=22))
        return p

    # ---- panel toggles ------------------------------------------------

    def _steam_ready(self):
        """Hosting and joining both need Steam - checked up front, before
        either panel opens, so nobody fills in a lobby's settings only to
        be told it can't be created."""
        if self.net.available:
            return True
        self.host_panel.visible = False
        self.join_panel.visible = False
        self._cancel_password()
        self.status.text = "Steam isn't running - start Steam, then restart the game"
        return False

    def toggle_host(self):
        if not self._steam_ready():
            return
        show = not self.host_panel.visible
        self.host_panel.visible = show
        self.join_panel.visible = False
        self.status.text = ""

    def toggle_join(self):
        if not self._steam_ready():
            return
        show = not self.join_panel.visible
        self.join_panel.visible = show
        self.host_panel.visible = False
        self.status.text = ""
        self._cancel_password()
        if show:
            self.refresh()

    def _on_players(self, value):
        self.players_label.text = f"Players: {int(value)}"

    # ---- host ---------------------------------------------------------

    def _create(self):
        if self._busy:
            return
        name = self.name_input.text.strip() or "RatWar Lobby"
        map_key = self.maps[self.map_dropdown.selected][1]
        self._busy = True
        self.status.text = "Creating lobby..."
        self.net.host_lobby(name, map_key, int(self.players_slider.value), self.pass_input.text,
                            lambda ok, info: self._on_hosted(ok, info, map_key))

    def _on_hosted(self, ok, info, map_key):
        self._busy = False
        if ok:
            self.on_start(map_key)
        else:
            self.status.text = f"Couldn't create lobby: {info}"

    # ---- join ---------------------------------------------------------

    def refresh(self):
        self._searching = True
        self._since_refresh = 0.0
        self._update_list_status()
        self.net.request_lobbies(self._on_lobbies)

    def _on_lobbies(self, lobbies, error):
        self._searching = False
        if error:
            self.status.text = f"Couldn't search for lobbies: {error}"
            self.lobbies = []
        else:
            self.lobbies = lobbies
        self._rebuild_rows()

    def _visible_lobbies(self):
        query = self.search_input.text.strip().lower()
        shown = [l for l in self.lobbies
                 if l["map"] in self._map_names and query in l["name"].lower()]
        return sorted(shown, key=lambda l: l["name"].lower())

    def _rebuild_rows(self):
        self.list.clear_children()
        for info in self._visible_lobbies():
            self.list.add(self._make_row(info))
        self._update_list_status()

    def _update_list_status(self):
        count = len(self.list.children)
        if self._searching and not count:
            self.list_status.text = "Searching..."
        elif not count:
            self.list_status.text = "No lobbies found" if not self.search_input.text.strip() \
                else "No lobbies match your search"
        else:
            self.list_status.text = f"{count} lobb{'y' if count == 1 else 'ies'}"

    def _make_row(self, info):
        full = info["max_players"] > 0 and info["players"] >= info["max_players"]
        color = _MUTED if full else (255, 255, 255, 255)
        row = Button("", on_click=lambda i=info: self._select(i), size=(0, 40), size_frac=(1, 0))
        row.add(Label(info["name"], font_size=22, color=color, anchor=(0, 0.5), offset=(12, 0)))
        row.add(Label(self._map_names[info["map"]], font_size=20, color=color,
                      anchor=(0.56, 0.5), pivot=(0, 0.5)))
        row.add(Label(f"{info['players']}/{info['max_players'] or '?'}", font_size=20, color=color,
                      anchor=(0.86, 0.5), pivot=(0, 0.5)))
        if info["has_password"]:
            row.add(Label("PW", font_size=18, color=_WARN, anchor=(1, 0.5), offset=(-8, 0)))
        return row

    def _select(self, info):
        if self._busy:
            return
        if 0 < info["max_players"] <= info["players"]:
            self.status.text = "That lobby is full"
            return
        self.status.text = ""
        if info["has_password"]:
            self._selected = info
            self.password_prompt.text = f"Password for {info['name']}"
            self.join_pass_input.set_text("", notify=False)
            self.password_panel.visible = True
            self.ui.set_focus(self.join_pass_input)
        else:
            self._join(info)

    def _submit_password(self):
        info = self._selected
        if info is None:
            return
        if self.net.check_password(info, self.join_pass_input.text):
            self._cancel_password()
            self._join(info)
        else:
            self.status.text = "Wrong password"
            self.join_pass_input.set_text("", notify=False)

    def _cancel_password(self):
        self._selected = None
        self.password_panel.visible = False
        self.ui.set_focus(None)

    def _join(self, info):
        self._busy = True
        self.status.text = f"Joining {info['name']}..."
        self.net.join_lobby(info["id"], lambda ok, res: self._on_joined(ok, res, info))

    def _on_joined(self, ok, result, info):
        self._busy = False
        if ok:
            self.on_start(info["map"])
        else:
            self.status.text = f"Couldn't join: {result}"

    # ---- per-frame ---------------------------------------------------

    def update(self, dt):
        """Auto-refreshes the lobby list while the Join panel is open (not
        mid-click, or a rebuilt row would swallow the press)."""
        if not self.root.visible or not self.join_panel.visible or self._busy \
                or self._searching or self.ui._pressed is not None:
            return
        self._since_refresh += dt
        if self._since_refresh >= self.REFRESH_SECONDS:
            self.refresh()
