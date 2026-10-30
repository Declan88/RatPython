"""
Controller navigation for the mouse-driven UI: moves the real (hidden) cursor onto the nearest control in a
direction, so the UI's own hover and click handling (A posts a click - Modules/Input) work unchanged. Left/right on
a slider steps its value instead of moving away; up/down in an open dropdown moves between its options; the left
stick on a colour wheel moves its marker.
"""

import math

import pygame

from .controls import ColorWheel, Slider

SLIDER_STEP = 0.05    # of the slider's range, per D-pad press


def _move_cursor(ui, x, y):
    pygame.mouse.set_pos((round(x * ui.scale), round(y * ui.scale)))


def hovered_wheel(ui):
    """The colour wheel under the cursor, if any (the left stick then drives its marker, not navigation)."""
    return ui._hover if isinstance(ui._hover, ColorWheel) else None


def drive_wheel(ui, wheel, stick, dt, speed):
    """Moves `wheel`'s marker by the left stick (`speed` radii/second at full tilt), and the cursor with it so a
    click (A) lands on the same colour. Returns whether the cursor moved."""
    sx, sy = stick
    if not sx and not sy:
        return False
    x, y, w, h = wheel.rect
    radius = min(w, h) / 2.0
    cx, cy = x + w / 2.0, y + h / 2.0
    angle = wheel.hue * 2.0 * math.pi
    mx = math.cos(angle) * wheel.saturation * radius + sx * speed * radius * dt
    my = -math.sin(angle) * wheel.saturation * radius + sy * speed * radius * dt
    distance = math.hypot(mx, my)
    if distance > radius:
        mx, my = mx / distance * radius, my / distance * radius
    wheel._set_from_point(cx + mx, cy + my)
    _move_cursor(ui, cx + mx * 0.99, cy + my * 0.99)    # just inside, so the cursor stays over the wheel
    return True


def _navigate_dropdown(ui, dropdown, direction):
    """Up/down between an open dropdown's options; left/right do nothing."""
    if direction not in ("up", "down"):
        return False
    mx, my = pygame.mouse.get_pos()
    y = my / ui.scale
    rows = dropdown.row_centres()
    if direction == "down":
        after = [r for r in rows if r[1] > y + 1.0]
        target = after[0] if after else None
    else:
        before = [r for r in rows if r[1] < y - 1.0]
        target = before[-1] if before else None
    if target is None:
        return False
    _move_cursor(ui, *target)
    return True


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
    if ui.popup is not None and hasattr(ui.popup, "row_centres"):
        return _navigate_dropdown(ui, ui.popup, direction)
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
    _move_cursor(ui, best[1], best[2])
    return True
