"""
Hidden-by-default demo of every anchor preset plus a centered menu using
the stacking layout and each widget type. app.py toggles it with F1 (which
also frees/recaptures the cursor).
"""

from .controls import Checkbox, ProgressBar, ScrollBox, Slider
from .widgets import Anchor, Button, Label, Panel

_ANCHORS = [
    ("TOP_LEFT", Anchor.TOP_LEFT), ("TOP_CENTER", Anchor.TOP_CENTER), ("TOP_RIGHT", Anchor.TOP_RIGHT),
    ("MIDDLE_LEFT", Anchor.MIDDLE_LEFT), ("MIDDLE_RIGHT", Anchor.MIDDLE_RIGHT),
    ("BOTTOM_LEFT", Anchor.BOTTOM_LEFT), ("BOTTOM_CENTER", Anchor.BOTTOM_CENTER),
    ("BOTTOM_RIGHT", Anchor.BOTTOM_RIGHT),
]


def build_demo(ui, on_close=None):
    """Adds the demo to ui.root (hidden) and returns its container."""
    demo = Panel(size_frac=(1, 1), visible=False, name="ui_demo")

    for name, anchor in _ANCHORS:
        # Inset from the anchored edge/corner so each tile sits inside it.
        inset = (16 * (1 - 2 * anchor[0]), 16 * (1 - 2 * anchor[1]))
        tile = demo.add(Panel(color=(20, 20, 30, 200), anchor=anchor, offset=inset, size=(190, 44)))
        tile.add(Label(name, font_size=22, anchor=Anchor.CENTER))

    menu = demo.add(Panel(color=(15, 15, 25, 225), anchor=Anchor.CENTER, layout="vertical",
                          spacing=12, padding=20, align="center", fit_content=True))
    menu.add(Label("UI DEMO", font_size=40))

    bar = menu.add(ProgressBar(value=0.65, label_format="{percent:.0f}%", size=(320, 26)))
    menu.add(Slider(value=0.65, step=0.05, size=(320, 28),
                    on_change=lambda v: setattr(bar, "value", v)))
    menu.add(Checkbox("Red bar", size=(320, 28), on_toggle=lambda on: setattr(
        bar, "fill_color", (220, 70, 70, 255) if on else (90, 200, 110, 255))))

    rows = menu.add(ScrollBox(size=(320, 150)))
    for i in range(1, 16):
        rows.add(Label(f"Scrollable row {i}  (wheel or drag the bar)", font_size=20))

    def close():
        demo.visible = False
        ui.set_cursor_free(False)
        if on_close is not None:
            on_close()

    menu.add(Button("Close (F1)", on_click=close))
    ui.root.add(demo)
    return demo
