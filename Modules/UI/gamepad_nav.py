"""
D-pad navigation for the mouse-driven UI: moves the real cursor onto the nearest control in a direction, so the
UI's own hover and click handling (A posts a click - Modules/Input) work unchanged. Left/right on a slider steps
its value instead of moving away.
"""

import pygame

from .controls import Slider

SLIDER_STEP = 0.05    # of the slider's range, per D-pad press


def _candidates(ui):
    """Every visible interactive control the cursor could select: hittable at its own centre (so a covered,
    clipped or scrolled-out one is skipped), not a whole scrolling container. Inside an open dropdown, only it."""
    found = []

    def walk(widget):
        if not widget.visible:
            return
        for child in widget.children:
            walk(child)
        if widget.interactive and not getattr(widget, "scrollable", False) and widget.rect[2] > 0 and widget.rect[3] > 0:
            x, y, w, h = widget.rect
            cx, cy = x + w / 2.0, y + h / 2.0
            if ui._pick(cx, cy) is widget:
                found.append((widget, cx, cy))

    walk(ui.popup if ui.popup is not None else ui.root)
    return found


def _score(dx, dy, direction):
    """How good a control at offset (dx, dy) is for `direction` - lower is better; None if it's the wrong way."""
    along, across = {"up": (-dy, dx), "down": (dy, dx), "left": (-dx, dy), "right": (dx, dy)}[direction]
    if along <= 1.0:
        return None
    return along + 2.5 * abs(across)        # prefer straight ahead over a closer one far off to the side


def navigate(ui, direction):
    """Moves the cursor to the best control in `direction` ("up", "down", "left", "right"). Returns whether the
    cursor moved (the caller tells the input manager, so the motion isn't taken for the mouse)."""
    mx, my = pygame.mouse.get_pos()
    x, y = mx / ui.scale, my / ui.scale
    current = ui._pick(x, y)
    if isinstance(current, Slider) and direction in ("left", "right"):
        span = current.max_value - current.min_value
        current.set_value(current.value + (SLIDER_STEP if direction == "right" else -SLIDER_STEP) * span)
        return False
    if current is not None:     # measure from the middle of the control we're on, not wherever on it the cursor is
        cx, cy, cw, ch = current.rect
        x, y = cx + cw / 2.0, cy + ch / 2.0
    best = None
    for widget, wx, wy in _candidates(ui):
        if widget is current:
            continue
        score = _score(wx - x, wy - y, direction)
        if score is not None and (best is None or score < best[0]):
            best = (score, wx, wy)
    if best is None and current is None:
        # Nothing selected yet and nothing that way: start on the control nearest the cursor.
        options = [(abs(wx - x) + abs(wy - y), wx, wy) for _, wx, wy in _candidates(ui)]
        best = min(options) if options else None
    if best is None:
        return False
    pygame.mouse.set_pos((round(best[1] * ui.scale), round(best[2] * ui.scale)))
    return True
