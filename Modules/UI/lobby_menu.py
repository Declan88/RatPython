"""
The main menu's Host / Join flow.

Host opens a settings panel (lobby name, map dropdown, game mode dropdown,
optional password, player count 2-10); Create asks NetworkManager to host
and then calls on_lobby_ready(map_key, mode_key). Join opens a searchable,
scrollable list of open lobbies (auto-refreshing while open); picking one -
after a password prompt if it has one - joins it and calls on_lobby_ready
with that lobby's own map/mode (read back from Steam lobby metadata - see
NetworkManager.host_lobby/_lobby_info - not from this client's own
dropdowns, which only exist in the Host panel).

on_lobby_ready, NOT on_start, is what actually fires on a successful host/
join now - it hands off to the waiting-room screen (Modules/UI/
lobby_room.py), which is what calls on_start later, once the host presses
Start there. _start_singleplayer is the one path that still calls on_start
directly (no lobby, no other players, nothing to wait for).

NetworkManager's on_result callbacks can run inside Steam's callback
dispatch, so everything they reach here (_on_lobbies/_on_hosted/_on_joined)
only updates widgets/flags - on_start/on_lobby_ready themselves are
expected to defer the real work (app.py just records a pending start/
pending lobby-room-entry for its main loop).
"""

from .controls import Checkbox, ColorWheel, ScrollBox, Slider
from .inputs import Dropdown, TextInput
from . import theme
from .widgets import Anchor, Button, Image, Label, Panel
from Modules.Player.hats import available_hats, display_name
from Modules.Player.rat_colors import hsv_to_rgb, rgb_to_hsv
from Modules.settings import settings

# Where the fur color's brightness slider starts (and returns to on reset) -
# well below full so a freshly picked color isn't blown out.
_DEFAULT_BRIGHTNESS = 0.45
_W = 700  # inner width of the host/join panels (736 - 2 * 18 padding)
_MUTED = theme.MUTED
_WARN = theme.WARN


def _button(text, on_click, **kw):
    return Button(text, on_click=on_click, **theme.BUTTON, **kw)


class LobbyMenu:
    REFRESH_SECONDS = 6.0
    MAX_PLAYERS = 10

    def __init__(self, ui, net, maps, modes, on_start, on_lobby_ready=None,
                 on_hat_change=None, on_color_change=None):
        """maps: [(display name, scene key), ...]; modes: [(display name,
        game mode id), ...] - see Modules.GameModes.registry; on_start(
        scene_key, mode_id); on_hat_change(hat name or None) fires when the
        wardrobe pick changes; on_color_change((r, g, b) 0-1 or None) when
        the fur color does.

        on_lobby_ready(map_key, mode_id): fires in place of on_start the
        instant hosting/joining a REAL lobby succeeds (see _on_hosted/
        _on_joined) - takes you to the waiting-room screen (Modules/UI/
        lobby_room.py) instead of straight into gameplay; on_start itself
        only fires later, once the host presses Start there. None (the
        default) falls back to on_start directly, same as before this
        existed - _start_singleplayer never goes through either of these,
        since there's no lobby/other players to wait for at all."""
        self.ui = ui
        self.net = net
        self.maps = maps
        self.modes = modes
        self._map_names = {key: name for name, key in maps}
        self.on_start = on_start
        self.on_lobby_ready = on_lobby_ready or on_start
        self.on_hat_change = on_hat_change
        self.on_color_change = on_color_change
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
        self._sidebar = sidebar     # hidden while the join panel is up - see _set_join_visible
        sidebar.add(Panel(color=theme.ACCENT, size=(4, 0), size_frac=(0, 1), anchor=Anchor.TOP_RIGHT))
        column = sidebar.add(Panel(anchor=Anchor.TOP_LEFT, offset=(80, 50), layout="vertical",
                                   spacing=14, fit_content=True))
        column.add(Label("RAT KING", font_size=132, color=theme.ACCENT, font=theme.title_font_path()))
        self._build_profile(column)
        column.add(Panel(size=(1, 18)))  # spacer
        # Multiplayer needs Steam; without it there's nothing HOST/JOIN could actually do
        # (see _steam_ready) - a single "Start singleplayer" button stands in for both,
        # straight into the main map with no lobby at all. Both live in the same slot,
        # toggled by _update_play_buttons rather than one replacing the other, so this
        # doesn't have to rebuild anything if Reconnect later succeeds mid-session.
        # Vertically centred stack. The status line and join list hang below it as their
        # own widgets (pivot at their top), so showing them never shifts the buttons.
        middle = sidebar.add(Panel(anchor=Anchor.MIDDLE_LEFT, offset=(80, 0), layout="vertical",
                                   spacing=14, fit_content=True))
        self.mp_buttons = middle.add(Panel(layout="vertical", spacing=14, fit_content=True))
        self.mp_buttons.add(_button("HOST", self.toggle_host, size=(_W + 36, 84), font_size=40))
        self.mp_buttons.add(_button("JOIN", self.toggle_join, size=(_W + 36, 84), font_size=40))
        self.sp_button = middle.add(_button(
            "START SINGLEPLAYER", self._start_singleplayer, size=(_W + 36, 84), font_size=36))
        self.status = sidebar.add(Label("", font_size=26, color=_WARN, anchor=Anchor.MIDDLE_LEFT,
                                        pivot=(0, 0), offset=(80, 105)))

        self._build_wardrobe(root)

        # Fullscreen join window, added last so it covers the sidebar and the rat preview.
        self.join_panel = root.add(Panel(color=theme.SIDEBAR, size_frac=(1, 1), visible=False))
        self.join_panel.add(self._build_join_panel())
        self.join_panel.add(_button("Back", self.toggle_join, size=(160, 60), font_size=30,
                                    anchor=Anchor.TOP_LEFT, offset=(40, 40)))
        self._update_play_buttons()     # needs join_panel (and the buttons above), so only now
        return root

    def _build_profile(self, column):
        """Steam identity: avatar + persona name once connected, or a warning plus a
        Reconnect button when the game started (or Steam closed) without it - see
        _rebuild_profile for why this is rebuilt outright on any state change rather than
        just toggling visibility (the two states share no widgets)."""
        self.profile_panel = column.add(Panel(layout="horizontal", spacing=12, fit_content=True))
        self._avatar_image = None
        self._avatar_loaded = False
        self._name_label = None
        self._rebuild_profile()

    def _rebuild_profile(self):
        self.profile_panel.clear_children()
        self._avatar_image = None
        self._avatar_loaded = False
        self._name_label = None
        if self.net.available:
            self._avatar_image = self.profile_panel.add(Image(size=(48, 48), anchor=Anchor.MIDDLE_LEFT))
            self._name_label = self.profile_panel.add(Label(
                self.net.local_name or "...", font_size=24, color=theme.TEXT, anchor=Anchor.MIDDLE_LEFT))
            self._apply_avatar()
        else:
            warn = self.profile_panel.add(Panel(layout="vertical", spacing=8, fit_content=True))
            warn.add(Label("Steam uninitialized! Open Steam and press reconnect",
                          font_size=20, color=_WARN))
            warn.add(_button("Reconnect", self._reconnect, size=(220, 44), font_size=22))

    def _apply_avatar(self):
        if self._avatar_image is None or self._avatar_loaded:
            return
        data = self.net.avatar_rgba(self.net.local_steam_id)
        if data is not None:
            self._avatar_image.set_rgba((64, 64), data)
            self._avatar_loaded = True

    def _reconnect(self):
        """The profile panel's Reconnect button: retries Steam from scratch (see
        NetworkManager.reconnect's own docstring) and, on success, swaps the whole menu
        over to the multiplayer-available state - the profile (now showing the avatar/
        name instead of the warning) and HOST/JOIN in place of Start Singleplayer."""
        if self.net.reconnect():
            self._rebuild_profile()
            self._update_play_buttons()

    def _update_play_buttons(self):
        available = self.net.available
        self.mp_buttons.visible = available
        self.sp_button.visible = not available
        if not available:
            self._set_join_visible(False)

    def _start_singleplayer(self):
        """Straight into the main map, no lobby, no Steam needed at all - see maps' own
        order (main_menu_scene.py builds this from app.py's MAPS, "Main Map" first).
        Defaults to "deathmatch" the same way - no mode picker exists outside the
        Host panel, and this path never touches it."""
        default_map = next((key for name, key in self.maps if key == "mainmap"), self.maps[0][1])
        default_mode = next((key for name, key in self.modes if key == "deathmatch"), self.modes[0][1])
        self.on_start(default_map, default_mode)

    def _build_wardrobe(self, root):
        """A card under the 3D rat with the hat dropdown. The choice lives on
        NetworkManager (local_hat): it's put on the local rat when a map
        starts, sent to every other player in each movement packet, and
        shown on the preview rat right away via on_hat_change."""
        hats = available_hats()
        stage = root.add(Panel(anchor=Anchor.TOP_RIGHT, size=(-theme.SIDEBAR_WIDTH, 0), size_frac=(1, 1)))
        self._stage = stage         # hidden while the join panel is up - see _set_join_visible
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
        self._build_color_card(stage)

    def _build_color_card(self, stage):
        """A card at the right of the 3D rat with the fur color wheel. Like the
        hat, the choice lives on NetworkManager (local_color): shown on the
        preview rat right away via on_color_change, put on the local rat when a
        map starts, and sent to everyone else in each movement packet."""
        card = stage.add(Panel(color=theme.CARD, anchor=Anchor.MIDDLE_RIGHT, pivot=(1.0, 0.5),
                              offset=(-40, 0), layout="vertical", spacing=12, padding=22,
                              align="center", fit_content=True, visible=False))
        self.color_card = card

        # The card starts closed; this palette button (top right of the stage)
        # opens/closes it.
        def toggle_card():
            card.visible = not card.visible
            if card.visible:
                self._set_settings_open(False)
            self.palette_button.normal_color = (theme.ACCENT_DIM if card.visible
                                                else theme.BUTTON["color"])

        self.palette_button = stage.add(Button(
            "", on_click=toggle_card, size=(80, 80), anchor=Anchor.TOP_RIGHT, offset=(-40, 40),
            **theme.BUTTON))
        self.palette_button.add(Image("Assets/Textures/UI/Palette.png", size=(56, 56),
                                      anchor=Anchor.CENTER))
        self._build_settings(stage)
        card.add(Label("FUR COLOR", font_size=28, color=theme.ACCENT))
        self.color_wheel = card.add(ColorWheel(value=_DEFAULT_BRIGHTNESS, size=(260, 260),
                                                  on_change=self._on_wheel))
        card.add(Label("Brightness", font_size=22, color=_MUTED))
        self.color_brightness = card.add(Slider(_DEFAULT_BRIGHTNESS, 0.15, 1.0, size=(260, 34),
                                                on_change=self._on_brightness))
        card.add(_button("Default colors", self._reset_color, size=(260, 54), font_size=26))
        current = self.net.local_color
        if current is not None:
            h, s, v = rgb_to_hsv(current)
            self.color_wheel.set_hs(h, s)
            self.color_wheel.set_value(v)
            self.color_brightness.set_value(v, notify=False)

    def _build_settings(self, stage):
        """A gear button beside the palette button that opens a card of graphics options
        (Modules/settings.py) - they take effect immediately and are saved."""
        card = stage.add(Panel(color=theme.CARD, anchor=Anchor.TOP_RIGHT, offset=(-40, 140),
                              layout="vertical", spacing=14, padding=22, fit_content=True,
                              visible=False))
        self.settings_card = card
        self.settings_button = stage.add(Button(
            "", on_click=lambda: self._set_settings_open(not card.visible), size=(80, 80),
            anchor=Anchor.TOP_RIGHT, offset=(-140, 40), **theme.BUTTON))
        self.settings_button.add(Image("Assets/Textures/UI/Settings.png", size=(56, 56),
                                       anchor=Anchor.CENTER))
        card.add(Label("SETTINGS", font_size=28, color=theme.ACCENT))

        def option(text, name):
            def changed(checked):
                setattr(settings, name, checked)
                settings.save()
            card.add(Checkbox(text, checked=getattr(settings, name), on_toggle=changed,
                              font_size=26, size=(380, 40)))

        option("Reflections (SSR)", "ssr")
        option("Gibs on death", "gibs")
        card.add(Label("Off: water uses a cheap reflection,", font_size=20, color=_MUTED))
        card.add(Label("deaths are a blood burst, not gibs.", font_size=20, color=_MUTED))

    def _set_settings_open(self, show):
        if not hasattr(self, "settings_card"):
            return
        self.settings_card.visible = show
        self.settings_button.normal_color = theme.ACCENT_DIM if show else theme.BUTTON["color"]
        if show and hasattr(self, "color_card") and self.color_card.visible:
            self.color_card.visible = False
            self.palette_button.normal_color = theme.BUTTON["color"]

    def _apply_color(self):
        w = self.color_wheel
        self.net.local_color = hsv_to_rgb(w.hue, w.saturation, w.value)
        if self.on_color_change is not None:
            self.on_color_change(self.net.local_color)

    def _on_wheel(self, hue, saturation):
        self._apply_color()

    def _on_brightness(self, value):
        self.color_wheel.set_value(value)
        self._apply_color()

    def _reset_color(self):
        self.net.local_color = None
        self.color_wheel.set_hs(0.0, 0.0)
        self.color_wheel.set_value(_DEFAULT_BRIGHTNESS)
        self.color_brightness.set_value(_DEFAULT_BRIGHTNESS, notify=False)
        if self.on_color_change is not None:
            self.on_color_change(None)

    def _panel(self):
        return Panel(color=theme.CARD, layout="vertical", spacing=10, padding=18,
                     fit_content=True, size=(_W + 36, 0), visible=False)

    def _build_join_panel(self):
        # Fills the fullscreen overlay (below the Back button), not _panel()'s fixed card.
        p = Panel(color=theme.CARD, layout="vertical", spacing=10, padding=18,
                  size=(-160, -240), size_frac=(1, 1), anchor=Anchor.CENTER)
        search = p.add(Panel(layout="horizontal", spacing=10, fit_content=True, size_frac=(1, 0)))
        self.search_input = search.add(TextInput(placeholder="Search lobby name...",
                                                 max_length=24, size=(-170, 54), size_frac=(1, 0),
                                                 font_size=26,
                                                 on_change=lambda _t: self._rebuild_rows()))
        search.add(_button("Refresh", self.refresh, size=(150, 54), font_size=26))

        # Header columns line up with _make_row's label anchors: the list's
        # rows sit 6px in (ScrollBox padding) and are 502 wide (524 minus
        # padding and the scrollbar).
        header = p.add(Panel(size=(-22, 26), size_frac=(1, 0), offset=(6, 0)))
        header.add(Label("Lobby", font_size=22, color=_MUTED, anchor=(0, 0.5), offset=(12, 0)))
        header.add(Label("Map", font_size=22, color=_MUTED, anchor=(0.56, 0.5), pivot=(0, 0.5)))
        header.add(Label("Players", font_size=22, color=_MUTED, anchor=(0.86, 0.5), pivot=(0, 0.5)))

        self.list = p.add(ScrollBox(size=(0, -150), size_frac=(1, 1), padding=6, spacing=8))
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

    def _set_join_visible(self, show):
        """Shows/hides the join window - and hides the rest of the menu (the sidebar's title and
        buttons, the wardrobe and fur/settings cards) while it is up, the way the lobby room
        hides the whole menu. The join window is translucent (so the 3D scene shows through),
        which would otherwise let all of that bleed through it."""
        self.join_panel.visible = show
        self._sidebar.visible = not show
        self._stage.visible = not show

    def _steam_ready(self):
        """Hosting and joining both need Steam - checked up front, before
        either panel opens, so nobody fills in a lobby's settings only to
        be told it can't be created."""
        if self.net.available:
            return True
        self._set_join_visible(False)
        self._cancel_password()
        self.status.text = "Steam isn't running - start Steam, then restart the game"
        return False

    def toggle_host(self):
        if not self._steam_ready():
            return
        self._set_join_visible(False)
        self._cancel_password()
        self.status.text = ""
        self._create()

    def toggle_join(self):
        if not self._steam_ready():
            return
        show = not self.join_panel.visible
        self._set_join_visible(show)
        self.status.text = ""
        self._cancel_password()
        if show:
            self.refresh()

    # ---- host ---------------------------------------------------------

    def _create(self):
        if self._busy:
            return
        name = "Rat King Lobby"
        map_key = next((key for _, key in self.maps if key == "mainmap"), self.maps[0][1])
        mode_key = next((key for _, key in self.modes if key == "cheese_war"), self.modes[0][1])
        self._busy = True
        self.status.text = "Creating lobby..."
        self.net.host_lobby(name, map_key, self.MAX_PLAYERS, "",
                            lambda ok, info: self._on_hosted(ok, info, map_key, mode_key),
                            mode_key=mode_key)

    def _on_hosted(self, ok, info, map_key, mode_key):
        self._busy = False
        if ok:
            self.status.text = ""
            self.on_lobby_ready(map_key, mode_key)
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
            self.list.size = (0, -256)  # make room for the prompt
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
        self.list.size = (0, -150)
        self.ui.set_focus(None)

    def _join(self, info):
        self._busy = True
        self.status.text = f"Joining {info['name']}..."
        self.net.join_lobby(info["id"], lambda ok, res: self._on_joined(ok, res, info))

    def _on_joined(self, ok, result, info):
        self._busy = False
        if ok:
            self.on_lobby_ready(info["map"], info["mode"])
        else:
            self.status.text = f"Couldn't join: {result}"

    # ---- per-frame ---------------------------------------------------

    def update(self, dt):
        """Auto-refreshes the lobby list while the Join panel is open (not
        mid-click, or a rebuilt row would swallow the press), and keeps the
        profile panel's name/avatar current - both load from Steam lazily
        (see NetworkManager._refresh_names/avatar_rgba's own docstrings), so
        neither is necessarily there yet the first time _rebuild_profile ran."""
        if self._name_label is not None:
            self._name_label.text = self.net.local_name or "..."
            self._apply_avatar()
        if not self.root.visible or not self.join_panel.visible or self._busy \
                or self._searching or self.ui._pressed is not None:
            return
        self._since_refresh += dt
        if self._since_refresh >= self.REFRESH_SECONDS:
            self.refresh()
