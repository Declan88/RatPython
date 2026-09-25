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
from . import theme
from .widgets import Anchor, Button, Label, Panel
from Modules.Player.hats import available_hats, display_name

_W = 700  # inner width of the host/join panels (736 - 2 * 18 padding)
_MUTED = theme.MUTED
_WARN = theme.WARN


def _button(text, on_click, **kw):
    return Button(text, on_click=on_click, **theme.BUTTON, **kw)


class LobbyMenu:
    REFRESH_SECONDS = 6.0
    MIN_PLAYERS = 2
    MAX_PLAYERS = 10

    def __init__(self, ui, net, maps, on_start, on_hat_change=None):
        """maps: [(display name, scene key), ...]; on_start(scene_key);
        on_hat_change(hat name or None) fires when the wardrobe pick changes."""
        self.ui = ui
        self.net = net
        self.maps = maps
        self._map_names = {key: name for name, key in maps}
        self.on_start = on_start
        self.on_hat_change = on_hat_change
        self.lobbies = []
        self._searching = False
        self._busy = False
        self._since_refresh = 0.0
        self._selected = None  # listed lobby waiting on a password
        self.root = self._build()

    # ---- construction ---------------------------------------------------

    def _build(self):
        root = Panel(size_frac=(1, 1), name="main_menu")

        # Left: a full-height sidebar with an accent edge holds the title and
        # the Host/Join flow; the right side is left open for the 3D rat.
        sidebar = root.add(Panel(color=theme.SIDEBAR, size=(theme.SIDEBAR_WIDTH, 0), size_frac=(0, 1),
                                 anchor=Anchor.TOP_LEFT))
        sidebar.add(Panel(color=theme.ACCENT, size=(4, 0), size_frac=(0, 1), anchor=Anchor.TOP_RIGHT))
        column = sidebar.add(Panel(anchor=Anchor.TOP_LEFT, offset=(80, 50), layout="vertical",
                                   spacing=14, fit_content=True))
        column.add(Label("RATWAR", font_size=132, color=theme.ACCENT, font=theme.title_font_path()))
        column.add(Panel(size=(1, 18)))  # spacer
        buttons = column.add(Panel(layout="horizontal", spacing=16, fit_content=True))
        buttons.add(_button("HOST", self.toggle_host, size=((_W + 36 - 16) // 2, 84), font_size=40))
        buttons.add(_button("JOIN", self.toggle_join, size=((_W + 36 - 16) // 2, 84), font_size=40))
        self.status = column.add(Label("", font_size=26, color=_WARN))
        self.host_panel = column.add(self._build_host_panel())
        self.join_panel = column.add(self._build_join_panel())

        self._build_wardrobe(root)
        return root

    def _build_wardrobe(self, root):
        """A card under the 3D rat with the hat dropdown. The choice lives on
        NetworkManager (local_hat): it's put on the local rat when a map
        starts, sent to every other player in each movement packet, and
        shown on the preview rat right away via on_hat_change."""
        hats = available_hats()
        stage = root.add(Panel(anchor=Anchor.TOP_RIGHT, size=(-theme.SIDEBAR_WIDTH, 0), size_frac=(1, 1)))
        card = stage.add(Panel(color=theme.CARD, anchor=(0.5, 0.84), pivot=(0.5, 0.0),
                              layout="vertical", spacing=10, padding=22, align="center",
                              fit_content=True))
        card.add(Label("WARDROBE", font_size=28, color=theme.ACCENT))

        def choose(index, _option):
            self.net.local_hat = hats[index - 1] if index > 0 else None
            if self.on_hat_change is not None:
                self.on_hat_change(self.net.local_hat)

        current = self.net.local_hat
        selected = hats.index(current) + 1 if current in hats else 0
        self.hat_dropdown = card.add(Dropdown(
            ["No hat"] + [display_name(h) for h in hats], selected=selected,
            on_change=choose, size=(400, 62), font_size=30, color=theme.BUTTON["color"],
            hover_color=theme.BUTTON["hover_color"], highlight_color=theme.ACCENT_DIM))

    def _panel(self):
        return Panel(color=theme.CARD, layout="vertical", spacing=10, padding=18,
                     fit_content=True, size=(_W + 36, 0), visible=False)

    def _build_host_panel(self):
        p = self._panel()
        p.add(Label("Lobby name", font_size=24, color=_MUTED))
        self.name_input = p.add(TextInput("RatWar Lobby", max_length=24, size=(_W, 54), font_size=26))
        p.add(Label("Map", font_size=24, color=_MUTED))
        self.map_dropdown = p.add(Dropdown([name for name, _ in self.maps], size=(_W, 54), font_size=26))
        p.add(Label("Password (optional)", font_size=24, color=_MUTED))
        self.pass_input = p.add(TextInput(password=True, placeholder="No password",
                                          max_length=24, size=(_W, 54), font_size=26))
        self.players_label = p.add(Label("", font_size=24, color=_MUTED))
        self.players_slider = p.add(Slider(4, self.MIN_PLAYERS, self.MAX_PLAYERS, step=1,
                                           size=(_W, 36), on_change=self._on_players))
        self._on_players(4)
        p.add(_button("Create lobby", self._create, size=(_W, 68), font_size=34))
        return p

    def _build_join_panel(self):
        p = self._panel()
        search = p.add(Panel(layout="horizontal", spacing=10, fit_content=True))
        self.search_input = search.add(TextInput(placeholder="Search lobby name...",
                                                 max_length=24, size=(_W - 160, 54), font_size=26,
                                                 on_change=lambda _t: self._rebuild_rows()))
        search.add(_button("Refresh", self.refresh, size=(150, 54), font_size=26))

        # Header columns line up with _make_row's label anchors: the list's
        # rows sit 6px in (ScrollBox padding) and are 502 wide (524 minus
        # padding and the scrollbar).
        header = p.add(Panel(size=(_W - 22, 26), offset=(6, 0)))
        header.add(Label("Lobby", font_size=22, color=_MUTED, anchor=(0, 0.5), offset=(12, 0)))
        header.add(Label("Map", font_size=22, color=_MUTED, anchor=(0.56, 0.5), pivot=(0, 0.5)))
        header.add(Label("Players", font_size=22, color=_MUTED, anchor=(0.86, 0.5), pivot=(0, 0.5)))

        self.list = p.add(ScrollBox(size=(_W, 330), padding=6, spacing=8))
        self.list_status = p.add(Label("", font_size=24, color=_MUTED))

        self.password_panel = p.add(Panel(layout="vertical", spacing=8, fit_content=True,
                                          visible=False))
        self.password_prompt = self.password_panel.add(Label("", font_size=24, color=_WARN))
        entry = self.password_panel.add(Panel(layout="horizontal", spacing=10, fit_content=True))
        self.join_pass_input = entry.add(TextInput(password=True, placeholder="Password",
                                                   max_length=24, size=(_W - 240, 54), font_size=26,
                                                   on_submit=lambda _t: self._submit_password()))
        entry.add(_button("Join", self._submit_password, size=(110, 54), font_size=26))
        entry.add(_button("Cancel", self._cancel_password, size=(110, 54), font_size=26))
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
        row = _button("", lambda i=info: self._select(i), size=(0, 52), size_frac=(1, 0))
        row.add(Label(info["name"], font_size=26, color=color, anchor=(0, 0.5), offset=(12, 0)))
        row.add(Label(self._map_names[info["map"]], font_size=24, color=color,
                      anchor=(0.56, 0.5), pivot=(0, 0.5)))
        row.add(Label(f"{info['players']}/{info['max_players'] or '?'}", font_size=24, color=color,
                      anchor=(0.86, 0.5), pivot=(0, 0.5)))
        if info["has_password"]:
            row.add(Label("PW", font_size=22, color=_WARN, anchor=(1, 0.5), offset=(-8, 0)))
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
