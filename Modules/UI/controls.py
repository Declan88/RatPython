"""
Stateful widgets built on the widget tree in widgets.py: ProgressBar,
Slider, Checkbox and ScrollBox. Interactive ones (Slider, Checkbox,
ScrollBox) use the mouse-capture protocol UIManager.handle_event drives:
on_press, then on_drag for every motion event while held, then on_release.
"""

from .widgets import Anchor, Label, Panel, Widget, _color


def _lerp_frac(value, lo, hi):
    return 0.0 if hi == lo else min(1.0, max(0.0, (value - lo) / (hi - lo)))


class ProgressBar(Widget):
    """A filled bar showing `value` within [min_value, max_value] - health,
    stamina, reload, a loading screen. Fills left-to-right ("horizontal")
    or bottom-to-top ("vertical"). label_format, if set, draws centered
    text formatted with value/min/max/percent (e.g. "{value:.0f}/{max:.0f}"
    or "{percent:.0f}%"); fill_color can be reassigned any time (e.g. to
    turn red when low)."""

    def __init__(self, value=1.0, min_value=0.0, max_value=1.0, orientation="horizontal",
                 color=(15, 15, 20, 200), fill_color=(90, 200, 110, 255),
                 label_format=None, font_size=18, **kw):
        kw.setdefault("size", (240, 22) if orientation == "horizontal" else (22, 160))
        super().__init__(**kw)
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.orientation = orientation
        self.color = color
        self.fill_color = fill_color
        self.label_format = label_format
        self.label = None
        if label_format is not None:
            self.label = self.add(Label("", font_size=font_size, anchor=Anchor.CENTER))

    @property
    def fraction(self):
        return _lerp_frac(self.value, self.min_value, self.max_value)

    def draw(self, out):
        x, y, w, h = self.rect
        out.rect(self.rect, _color(self.color))
        f = self.fraction
        if f > 0.0:
            if self.orientation == "horizontal":
                out.rect((x, y, w * f, h), _color(self.fill_color))
            else:
                out.rect((x, y + h * (1 - f), w, h * f), _color(self.fill_color))
        if self.label is not None:
            self.label.text = self.label_format.format(
                value=self.value, min=self.min_value, max=self.max_value, percent=f * 100.0)
        super().draw(out)


class Slider(Widget):
    """Horizontal draggable value in [min_value, max_value], optionally
    snapped to `step`. on_change(value) fires whenever the value changes."""

    interactive = True
    KNOB_WIDTH = 14.0

    def __init__(self, value=0.5, min_value=0.0, max_value=1.0, step=None, on_change=None,
                 track_color=(255, 255, 255, 60), fill_color=(90, 160, 255, 255),
                 knob_color=(225, 225, 232, 255), knob_active_color=(255, 255, 255, 255), **kw):
        kw.setdefault("size", (240, 28))
        super().__init__(**kw)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.on_change = on_change
        self.track_color = track_color
        self.fill_color = fill_color
        self.knob_color = knob_color
        self.knob_active_color = knob_active_color
        self.value = None
        self.set_value(value, notify=False)

    def set_value(self, value, notify=True):
        value = min(self.max_value, max(self.min_value, value))
        if self.step:
            value = self.min_value + round((value - self.min_value) / self.step) * self.step
            value = min(self.max_value, max(self.min_value, value))
        changed = value != self.value
        self.value = value
        if changed and notify and self.on_change is not None:
            self.on_change(value)

    def _track(self):
        x, _, w, _ = self.rect
        inset = self.KNOB_WIDTH / 2
        return x + inset, w - 2 * inset

    def _set_from_x(self, x):
        x0, tw = self._track()
        f = 0.0 if tw <= 0 else (x - x0) / tw
        self.set_value(self.min_value + f * (self.max_value - self.min_value))

    def on_press(self, x, y):
        self._set_from_x(x)

    def on_drag(self, x, y):
        self._set_from_x(x)

    def on_release(self, x, y, inside):
        pass

    def draw(self, out):
        x, y, w, h = self.rect
        x0, tw = self._track()
        f = _lerp_frac(self.value, self.min_value, self.max_value)
        kx = x0 + tw * f
        bar_h = min(6.0, h)
        by = y + (h - bar_h) / 2
        out.rect((x0, by, tw, bar_h), _color(self.track_color))
        out.rect((x0, by, kx - x0, bar_h), _color(self.fill_color))
        active = self.hovered or self.pressed
        out.rect((kx - self.KNOB_WIDTH / 2, y, self.KNOB_WIDTH, h),
                 _color(self.knob_active_color if active else self.knob_color))
        super().draw(out)


class Checkbox(Widget):
    """A toggle with a text label to its right. on_toggle(checked) fires on
    every change."""

    interactive = True

    def __init__(self, text="", checked=False, on_toggle=None, font_size=22,
                 box_color=(30, 30, 40, 230), hover_color=(60, 60, 78, 240),
                 check_color=(90, 200, 110, 255), **kw):
        kw.setdefault("size", (240, 28))
        super().__init__(**kw)
        self.checked = checked
        self.on_toggle = on_toggle
        self.box_color = box_color
        self.hover_color = hover_color
        self.check_color = check_color
        self.label = self.add(Label(text, font_size=font_size, anchor=Anchor.MIDDLE_LEFT,
                                    offset=(self.size[1] + 10, 0)))

    def click(self):
        self.checked = not self.checked
        if self.on_toggle is not None:
            self.on_toggle(self.checked)

    def draw(self, out):
        x, y, w, h = self.rect
        out.rect((x, y, h, h), _color(self.hover_color if self.hovered else self.box_color))
        if self.checked:
            pad = h * 0.25
            out.rect((x + pad, y + pad, h - 2 * pad, h - 2 * pad), _color(self.check_color))
        super().draw(out)


class ScrollBox(Panel):
    """A vertically scrolling list: children are stacked top-to-bottom
    (spacing/padding/align like a vertical Panel), clipped to the box, and
    scrolled with the mouse wheel or by dragging/clicking the scrollbar.
    Children can be anything - labels, buttons, whole sub-panels - and keep
    working while scrolled (hit-testing respects the clip)."""

    interactive = True
    scrollable = True

    def __init__(self, color=(0, 0, 0, 120), spacing=6, padding=6, align="start",
                 scrollbar_width=10, **kw):
        kw.setdefault("size", (300, 200))
        super().__init__(color=color, layout="vertical", spacing=spacing,
                         padding=padding, align=align, **kw)
        self.scrollbar_width = scrollbar_width
        self.scroll = 0.0
        self.content_height = 0.0
        self._grab = None  # y offset within the thumb while it's being dragged

    def _viewport(self):
        x, y, w, h = self._content_rect()
        return x, y, max(0.0, w - self.scrollbar_width), h

    def _max_scroll(self):
        return max(0.0, self.content_height - self._viewport()[3])

    def _arrange_children(self):
        vx, vy, vw, vh = self._viewport()
        kids = [c for c in self.children if c.visible]
        sizes = [c.measure(vw, vh) for c in kids]
        self.content_height = sum(s[1] for s in sizes) + self.spacing * max(0, len(kids) - 1)
        self.scroll = min(max(self.scroll, 0.0), self._max_scroll())
        pos = 0.0
        for c, (w, h) in zip(kids, sizes):
            shift = {"start": 0.0, "center": (vw - w) / 2, "end": vw - w}[self.align]
            c.arrange(vx + shift + c.offset[0], vy - self.scroll + pos + c.offset[1], w, h)
            pos += h + self.spacing

    def scroll_by(self, dy):
        self.scroll = min(max(self.scroll + dy, 0.0), self._max_scroll())

    def _thumb(self):
        """(track_x, thumb_y, thumb_h), or None when nothing overflows."""
        vx, vy, vw, vh = self._viewport()
        max_scroll = self._max_scroll()
        if max_scroll <= 0.0:
            return None
        th = max(24.0, vh * vh / self.content_height)
        return vx + vw, vy + (vh - th) * (self.scroll / max_scroll), th

    def pick(self, x, y):
        # Children scrolled out of view still have rects out there - only
        # let anything (them or the box itself) be hit inside the box.
        return super().pick(x, y) if self._contains(x, y) else None

    def pick_scroll(self, x, y):
        if not self._contains(x, y):
            return None
        return super().pick_scroll(x, y) or self

    def on_press(self, x, y):
        thumb = self._thumb()
        if thumb is None or x < thumb[0]:
            return
        _, ty, th = thumb
        if ty <= y < ty + th:
            self._grab = y - ty
        else:  # click on the track: page toward the click
            self.scroll_by(self._viewport()[3] * (1 if y > ty else -1))

    def on_drag(self, x, y):
        thumb = self._thumb()
        if self._grab is None or thumb is None:
            return
        _, vy, _, vh = self._viewport()
        travel = vh - thumb[2]
        if travel > 0:
            self.scroll = min(max((y - self._grab - vy) / travel, 0.0), 1.0) * self._max_scroll()

    def on_release(self, x, y, inside):
        self._grab = None

    def draw(self, out):
        if self.color[3] > 0:
            out.rect(self.rect, _color(self.color))
        out.push_clip(self._viewport())
        for c in self.children:
            if c.visible:
                c.draw(out)
        out.pop_clip()
        thumb = self._thumb()
        if thumb is not None:
            tx, ty, th = thumb
            _, vy, _, vh = self._viewport()
            out.rect((tx, vy, self.scrollbar_width, vh), _color((255, 255, 255, 25)))
            active = self._grab is not None or self.hovered
            out.rect((tx + 2, ty, self.scrollbar_width - 4, th),
                     _color((215, 215, 225, 200 if active else 130)))
