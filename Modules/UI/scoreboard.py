"""
Hold-Tab player list: a card listing everyone in the match with their Steam
name and profile picture. Built from ordinary UI widgets like the name tags;
rows are only rebuilt when a name or picture actually changes.
"""

from . import theme
from .widgets import Anchor, Image, Label, Panel

ROW_SIZE = (520, 60)
AVATAR_SIZE = 48
AVATAR_PIXELS = (64, 64)   # what NetworkManager.avatar_rgba hands back


class Scoreboard:
    def __init__(self, ui):
        self.ui = ui
        self.card = ui.root.add(Panel(color=theme.CARD, anchor=Anchor.TOP_CENTER, pivot=(0.5, 0.0),
                                      offset=(0, 140), layout="vertical", spacing=8, padding=18,
                                      fit_content=True, visible=False, name="scoreboard"))
        self._signature = None

    @property
    def visible(self):
        return self.card.visible

    @visible.setter
    def visible(self, value):
        self.card.visible = value

    def update(self, net_mgr):
        """Call each frame while visible. Rows: you first, then everyone else."""
        entries = [(net_mgr.local_steam_id, net_mgr.local_name or "You", True)]
        for steam_id, player in sorted(net_mgr.remote_players.items()):
            entries.append((steam_id, player.name or f"Player {steam_id % 10000}", False))
        rows = [(i, name, mine, net_mgr.avatar_rgba(i)) for i, name, mine in entries]

        signature = tuple((i, name, avatar is not None) for i, name, _, avatar in rows)
        if signature == self._signature:
            return
        self._signature = signature
        self.card.clear_children()
        self.card.add(Label(f"Players ({len(rows)})", font_size=26, color=theme.ACCENT))
        for _, name, mine, avatar in rows:
            self.card.add(self._row(name, mine, avatar))

    def _row(self, name, mine, avatar):
        row = Panel(color=(34, 38, 56, 240) if mine else (24, 27, 42, 240), layout="horizontal",
                    spacing=14, padding=6, align="center", size=ROW_SIZE)
        if avatar:
            pic = Image(size=(AVATAR_SIZE, AVATAR_SIZE))
            pic.set_rgba(AVATAR_PIXELS, avatar)
            row.add(pic)
        else:
            row.add(Panel(color=theme.CARD_BORDER, size=(AVATAR_SIZE, AVATAR_SIZE)))
        row.add(Label(name, font_size=26, color=theme.TEXT))
        if mine:
            row.add(Label("(you)", font_size=20, color=theme.MUTED))
        return row
