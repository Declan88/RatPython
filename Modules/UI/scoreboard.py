"""
Hold-Tab player list: a card listing everyone in the match with their Steam
name and profile picture. Built from ordinary UI widgets like the name tags;
rows are only rebuilt when a name or picture actually changes.
"""

from . import theme
from .widgets import Anchor, Image, Label, Panel

STAT_COLUMN_WIDTH = 56     # each of the K / D / KD columns - a fixed-width box,
STAT_COLUMN_GAP = 10       # not just a right-anchor point (see _stat_box's own
STAT_INSET = 10            # comment for why that alone looked uneven/ugly)
STAT_FONT_SIZE = 22
AVATAR_SIZE = 48
AVATAR_PAD = 6
AVATAR_PIXELS = (64, 64)   # what NetworkManager.avatar_rgba hands back

# A FIXED budget for the name (+ "(you)" tag) column, rather than however much
# space happens to be left over - see _row's own comment on why: reserving
# "enough for the widest plausible name" left a wide gap after every SHORT
# one, which is what actually looked wrong. A name that doesn't fit this
# budget at NAME_FONT_SIZE is shrunk to fit (see _fit_width) instead of
# growing the column.
NAME_COLUMN_WIDTH = 190
NAME_FONT_SIZE = 24
NAME_MIN_FONT_SIZE = 14
YOU_TAG_WIDTH = 56    # reserved out of NAME_COLUMN_WIDTH for "(you)", mine only

NAME_X = AVATAR_PAD + AVATAR_SIZE + 14
_STAT_AREA_WIDTH = STAT_INSET + 3 * STAT_COLUMN_WIDTH + 2 * STAT_COLUMN_GAP
ROW_WIDTH = NAME_X + NAME_COLUMN_WIDTH + _STAT_AREA_WIDTH
ROW_SIZE = (ROW_WIDTH, 60)

# Right-edge offset of each stat column's own box (KD closest to the row's
# edge, then D, then K) - shared by the header row and every player row so
# their columns land in exactly the same place regardless of how wide the
# name ends up. Each box is a fixed STAT_COLUMN_WIDTH regardless of its
# text's own length (see _stat_box) - a bare right-anchored label here (the
# very first version of this) put a 1-digit "3" and a 2-digit "12" at the
# same RIGHT edge, so the gap in front of each varied with its digit count -
# visually uneven, which is what actually looked ugly. A fixed box with the
# text CENTERED in it fixes that: every column keeps the same pitch no
# matter what's actually printed in it.
_KD_OFFSET = -STAT_INSET
_D_OFFSET = _KD_OFFSET - STAT_COLUMN_WIDTH - STAT_COLUMN_GAP
_K_OFFSET = _D_OFFSET - STAT_COLUMN_WIDTH - STAT_COLUMN_GAP
# See the nudge comment at its own two call sites in _row.
_VALUE_NUDGE = 4


def _kd_ratio(kills, deaths):
    return f"{kills / deaths:.2f}" if deaths > 0 else f"{kills:.2f}"


def _text_width(manager, text, font_size):
    """Logical-px width of text at font_size, via the same font the Label
    itself will render with (manager.get_font measures in DEVICE px - see
    Label._font_px - hence dividing back out by manager.scale)."""
    if manager is None:
        return 0.0
    px = max(1, round(font_size * manager.scale))
    return manager.get_font(None, px).size(text)[0] / max(manager.scale, 1e-6)


def _line_height(manager, font_size):
    if manager is None:
        return font_size
    px = max(1, round(font_size * manager.scale))
    return manager.get_font(None, px).get_linesize() / max(manager.scale, 1e-6)


def _stat_box(text, color, font_size, right_offset, manager, box_width=STAT_COLUMN_WIDTH):
    """A fixed-width, height-matched-to-text box whose text is CENTERED
    within it (align="center") and whose RIGHT edge sits at right_offset -
    see the module-level comment above _KD_OFFSET for why a fixed box
    instead of a bare right-anchored label. size is set explicitly (rather
    than left at Label's own auto-measure default), so its height must be
    supplied too, or Y-centering (anchor/pivot (·, 0.5)) would center
    around a 0-tall box and draw the text too high relative to a normal
    auto-sized Label (e.g. the name label) in the same row."""
    height = _line_height(manager, font_size)
    return Label(text, font_size=font_size, color=color, align="center",
                size=(box_width, height), anchor=(1, 0.5), pivot=(1, 0.5), offset=(right_offset, 0))


def _fit_font_size(manager, text, max_width, base_size=NAME_FONT_SIZE, min_size=NAME_MIN_FONT_SIZE):
    """base_size, or smaller (down to min_size) if text is too wide for
    max_width at base_size - font width scales ~linearly with px size for a
    fixed string, so one proportional step lands close enough (a font is
    also re-rendered/cached per (text, size) pair regardless - see Label._
    ensure_texture - so this doesn't cost more than any other size change)."""
    width = _text_width(manager, text, base_size)
    if width <= max_width or width <= 0:
        return base_size
    return max(min_size, base_size * max_width / width)


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

    def update(self, net_mgr, game_mode=None):
        """Call each frame while visible. Rows: you first, then everyone
        else, each with its kill/death tally - from game_mode.standings()
        if a GameMode is active (see Modules/GameModes/game_mode.py - this
        is how a future mode with different scoring shows up here without
        this file changing), or read directly off net_mgr/remote_players
        otherwise (kept working with no GameMode wired up at all)."""
        if game_mode is not None:
            standings = game_mode.standings()
        else:
            standings = [(net_mgr.local_steam_id, net_mgr.local_name or "You",
                         net_mgr.local_kills, net_mgr.local_deaths)]
            for steam_id, player in net_mgr.remote_players.items():
                standings.append((steam_id, player.name or f"Player {steam_id % 10000}",
                                  player.kills, player.deaths))
            standings.sort(key=lambda row: (-row[2], row[3]))
        rows = [
            (i, name, i == net_mgr.local_steam_id, kills, deaths, net_mgr.avatar_rgba(i))
            for i, name, kills, deaths in standings
        ]

        signature = tuple((i, name, kills, deaths, avatar is not None) for i, name, _, kills, deaths, avatar in rows)
        if signature == self._signature:
            return
        self._signature = signature
        self.card.clear_children()
        manager = self.card.manager
        self.card.add(Label(f"Players ({len(rows)})", font_size=26, color=theme.ACCENT))
        # Same width and right-edge offsets as each row below's stat boxes
        # (see _row/_stat_box), so K/D/KD sit directly above the values they
        # label, each centered over its own column rather than at its edge.
        header = self.card.add(Panel(size=(ROW_SIZE[0], 22)))
        header.add(_stat_box("K", theme.MUTED, 16, _K_OFFSET, manager))
        header.add(_stat_box("D", theme.MUTED, 16, _D_OFFSET, manager))
        header.add(_stat_box("KD", theme.MUTED, 16, _KD_OFFSET, manager))
        for _, name, mine, kills, deaths, avatar in rows:
            self.card.add(self._row(name, mine, kills, deaths, avatar))

    def _row(self, name, mine, kills, deaths, avatar):
        # NOT layout="horizontal" - that stacks children left-to-right at
        # explicit positions and ignores their own anchor entirely (see
        # Panel._arrange_children), which is exactly what a right-aligned K/D
        # column needs. Plain absolute anchor placement instead, same as the
        # header row above already uses successfully.
        row = Panel(color=(34, 38, 56, 240) if mine else (24, 27, 42, 240),
                    padding=6, size=ROW_SIZE)
        if avatar:
            pic = row.add(Image(size=(AVATAR_SIZE, AVATAR_SIZE), anchor=(0, 0.5), offset=(AVATAR_PAD, 0)))
            pic.set_rgba(AVATAR_PIXELS, avatar)
        else:
            row.add(Panel(color=theme.CARD_BORDER, size=(AVATAR_SIZE, AVATAR_SIZE),
                          anchor=(0, 0.5), offset=(AVATAR_PAD, 0)))

        manager = self.card.manager
        name_budget = NAME_COLUMN_WIDTH - (YOU_TAG_WIDTH if mine else 0)
        name_size = _fit_font_size(manager, name, name_budget)
        row.add(Label(name, font_size=name_size, color=theme.TEXT, anchor=(0, 0.5), offset=(NAME_X, 0)))
        if mine:
            # Right after the name's OWN (possibly shrunk) width, not a fixed
            # offset from it - so short names don't leave a gap before "(you)".
            name_w = _text_width(manager, name, name_size)
            row.add(Label("(you)", font_size=18, color=theme.MUTED, anchor=(0, 0.5),
                          offset=(NAME_X + name_w + 8, 0)))
        # A few px right of the header's own centering (_VALUE_NUDGE) - centering
        # by ADVANCE width (what font.size()/this centers on) still left digit
        # glyphs reading slightly left of the letter labeling them above, a
        # numeral's own side bearings being narrower than a capital letter's -
        # confirmed by pixel-measuring rendered text, not just eyeballing it.
        row.add(_stat_box(str(kills), theme.TEXT, STAT_FONT_SIZE, _K_OFFSET + _VALUE_NUDGE, manager))
        row.add(_stat_box(str(deaths), theme.MUTED, STAT_FONT_SIZE, _D_OFFSET + _VALUE_NUDGE, manager))
        row.add(_stat_box(_kd_ratio(kills, deaths), theme.ACCENT, STAT_FONT_SIZE, _KD_OFFSET + _VALUE_NUDGE, manager))
        return row
