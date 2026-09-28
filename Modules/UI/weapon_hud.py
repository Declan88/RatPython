"""
Bottom-right weapon HUD: the current weapon's icon, name and ammo, with a
strip of every other weapon slot above it (dim; the active one bright with
an accent border) so it's clear there's something to scroll to - see
app.py's own scroll-wheel weapon-switching.
"""

from . import theme
from .widgets import Anchor, Image, Label, Panel

ICON_SIZE = 60
SLOT_ICON_SIZE = 30
SLOT_BORDER = 3
DIM_TINT = (160, 160, 160, 140)


def _set_image(img, path):
    """Swaps an Image widget's source texture - Image itself only ever loads
    once (see its own _load's `if self._tex is None` guard), so changing
    .path alone would do nothing; releasing the old texture first (the same
    thing Image.set_rgba does internally) makes the next draw pick up the
    new one."""
    if img.path == path:
        return
    img._release_resources()
    img.path = path


class WeaponHUD(Panel):
    def __init__(self, **kw):
        kw.setdefault("anchor", Anchor.BOTTOM_RIGHT)
        kw.setdefault("pivot", (1.0, 1.0))
        kw.setdefault("offset", (-24, -24))
        super().__init__(color=(0, 0, 0, 0), visible=False, layout="vertical",
                         spacing=8, align="end", fit_content=True, **kw)
        # Row of every slot's icon - rebuilt only when the slot LIST itself
        # changes (which weapon is active changes far more often - see
        # update's own signature check - so this is split from the row
        # below it, which is cheap to refresh every frame).
        self.slots_row = self.add(Panel(layout="horizontal", spacing=8, fit_content=True, align="end"))
        self._slot_frames = []   # [(frame Panel, icon Image)], one per weapon slot
        self._slots_key = None

        self.card = self.add(Panel(color=theme.CARD, layout="horizontal", spacing=14,
                                   padding=14, align="center", fit_content=True))
        self.icon = self.card.add(Image(size=(ICON_SIZE, ICON_SIZE)))
        text_col = self.card.add(Panel(layout="vertical", spacing=2, fit_content=True, align="end"))
        self.name_label = text_col.add(Label("", font_size=20, color=theme.MUTED))
        self.ammo_label = text_col.add(Label("", font_size=34, color=theme.TEXT))

    def _rebuild_slots(self, slots):
        self.slots_row.clear_children()
        self._slot_frames = []
        for weapon in slots:
            # An accent-colored frame a few px bigger than the icon reads as
            # a border (Panel has no border concept of its own - see
            # inputs.TextInput's own focus outline for the same 4-rect
            # trick, not worth pulling in here for one square icon).
            frame = self.slots_row.add(Panel(color=(0, 0, 0, 0), padding=SLOT_BORDER,
                                             size=(SLOT_ICON_SIZE + 2 * SLOT_BORDER,) * 2))
            icon = frame.add(Image(getattr(weapon, "icon", None),
                                   size=(SLOT_ICON_SIZE, SLOT_ICON_SIZE), tint=DIM_TINT))
            self._slot_frames.append((frame, icon))

    def update(self, weapon, slots, current_index):
        """Call once a frame while a weapon's actually out. Doesn't touch
        .visible itself - app.py shows/hides this alongside the crosshair
        (dead, paused, menu...), same as every other HUD element."""
        key = tuple(id(w) for w in slots)
        if key != self._slots_key:
            self._slots_key = key
            self._rebuild_slots(slots)
        for i, (frame, icon) in enumerate(self._slot_frames):
            active = i == current_index
            frame.color = theme.ACCENT if active else (0, 0, 0, 0)
            icon.tint = (255, 255, 255, 255) if active else DIM_TINT
        # Only one slot: nothing to switch between, so the strip just adds
        # clutter - hide it.
        self.slots_row.visible = len(slots) > 1

        _set_image(self.icon, getattr(weapon, "icon", None))
        self.name_label.text = weapon.name
        if weapon.reloading:
            self.ammo_label.text = "RELOADING"
            self.ammo_label.color = theme.WARN
        elif weapon.magazine_size > 0:
            self.ammo_label.text = f"{weapon.ammo} / {weapon.magazine_size}"
            self.ammo_label.color = theme.ACCENT if weapon.ammo > 0 else (220, 70, 70, 255)
        else:
            self.ammo_label.text = ""
