"""
A CS:GO-style kill feed, top-left: a short-lived stack of "Killer [weapon
icon] [skull if headshot] Victim" rows, newest on top, each auto-expiring a
few seconds after it appears. Driven by NetworkManager.on_killfeed, which
fires for EVERY kill this client learns about (see that module's own
_broadcast_kill/_receive_killfeed) - not just the local player's own - so
every player sees the same feed, exactly like a real match scoreboard/feed.
"""

import time

from . import theme
from .widgets import Anchor, Image, Label, Panel

ROW_HEIGHT = 48
ROW_SPACING = 6
ICON_SIZE = 32
NAME_FONT_SIZE = 26
MAX_ROWS = 5   # older rows past this are dropped immediately, not just faded
PANEL_WIDTH = 460

def weapon_icon(weapon_name):
    """The icon file of the registered weapon called `weapon_name` (WeaponsBase.name, e.g.
    "USP"), or None (just the names are shown) for an unknown weapon or one with no icon."""
    from Modules.Weapons.registry import get_weapon_class, weapon_ids
    for weapon_id in weapon_ids():
        cls = get_weapon_class(weapon_id)
        if cls.name == weapon_name:
            return cls.icon
    return None


HEADSHOT_ICON = "Assets/Textures/Icons/Headshot/headshot.png"


class KillFeed(Panel):
    DISPLAY_SECONDS = 5.0
    FADE_SECONDS = 0.6

    def __init__(self, **kw):
        kw.setdefault("anchor", Anchor.TOP_LEFT)
        kw.setdefault("offset", (16, 16))
        kw.setdefault("size", (PANEL_WIDTH, MAX_ROWS * (ROW_HEIGHT + ROW_SPACING)))
        super().__init__(color=(0, 0, 0, 0), **kw)
        self._entries = []   # [{"start": t, "row": Panel, "labels": [...]}], newest first

    def add_kill(self, killer_name, victim_name, weapon_name, headshot=False,
                 killer_mine=False, victim_mine=False):
        """Adds a new row at the top, pushing existing ones down, and drops
        the oldest if that puts more than MAX_ROWS on screen (see module
        docstring - CS:GO does the same: a feed this busy is illegible past a
        handful of rows anyway, so the extra rows aren't worth keeping)."""
        row, labels = self._build_row(killer_name, victim_name, weapon_name, headshot, killer_mine, victim_mine)
        self.add(row)
        self._entries.insert(0, {"start": time.perf_counter(), "row": row, "labels": labels})
        while len(self._entries) > MAX_ROWS:
            self._remove(self._entries.pop())
        self._reflow()

    def update(self):
        """Call once a frame (app.py's main loop does, unconditionally - see
        DeathScreen/Hitmarker for the same "cheap no-op while nothing's
        showing" shape): fades out and drops entries past DISPLAY_SECONDS."""
        if not self._entries:
            return
        now = time.perf_counter()
        total = self.DISPLAY_SECONDS + self.FADE_SECONDS
        changed = False
        for entry in list(self._entries):
            elapsed = now - entry["start"]
            if elapsed >= total:
                self._entries.remove(entry)
                self._remove(entry["row"])
                changed = True
                continue
            alpha = 255
            if elapsed > self.DISPLAY_SECONDS:
                alpha = round(255 * (1.0 - (elapsed - self.DISPLAY_SECONDS) / self.FADE_SECONDS))
            entry["row"].color = (*entry["row"].color[:3], round(140 * alpha / 255))
            for label in entry["labels"]:
                label.color = (*label.color[:3], round(label.base_alpha * alpha / 255))
        if changed:
            self._reflow()

    def _remove(self, row):
        if row in self.children:
            self.remove(row)

    def _reflow(self):
        for i, entry in enumerate(self._entries):
            entry["row"].offset = (0, i * (ROW_HEIGHT + ROW_SPACING))

    def _build_row(self, killer_name, victim_name, weapon_name, headshot, killer_mine, victim_mine):
        row = Panel(color=(10, 11, 18, 160), layout="horizontal", spacing=10, padding=8,
                    align="center", size=(0, ROW_HEIGHT), size_frac=(1, 0))
        labels = []

        def name_label(text, mine):
            color = theme.ACCENT if mine else theme.TEXT
            lbl = row.add(Label(text, font_size=NAME_FONT_SIZE, color=color, shadow=True))
            lbl.base_alpha = color[3]
            labels.append(lbl)
            return lbl

        name_label(killer_name, killer_mine)
        icon = weapon_icon(weapon_name)
        if icon:
            row.add(Image(icon, size=(ICON_SIZE, ICON_SIZE)))
        if headshot:
            row.add(Image(HEADSHOT_ICON, size=(ICON_SIZE, ICON_SIZE)))
        name_label(victim_name, victim_mine)
        return row, labels
