"""
Keyboard/choice widgets: TextInput and Dropdown.

TextInput takes keyboard focus through UIManager.set_focus (click it, or
call manager.set_focus(widget)); while focused the manager routes every
KEYDOWN/TEXTINPUT event to on_key/on_text and swallows it. Dropdown
registers itself as manager.popup while open so its option list - drawn in
the deferred overlay pass, above every other widget - also wins hit tests.
"""

import time

import pygame

from .widgets import Anchor, Label, Widget, _color

_WHITE = (255, 255, 255, 255)


class TextInput(Widget):
    """Single-line editable text: caret (click to place, arrows/Home/End,
    Backspace/Delete), horizontal scroll to keep the caret visible,
    max_length, optional password masking, and a placeholder shown while
    empty and unfocused. on_change(text) fires on every edit and
    on_submit(text) on Enter.

    Selection: drag with the mouse, Shift+Left/Right/Home/End, or Ctrl+A;
    typing or Backspace/Delete replaces/removes it. Holding Backspace (or
    Delete/arrows) repeats - UIManager.set_focus turns on key repeat only
    while a TextInput has focus. Not supported: clipboard."""

    interactive = True
    PAD = 10.0

    def __init__(self, text="", placeholder="", password=False, max_length=64,
                 on_change=None, on_submit=None, font_size=22,
                 color=(20, 20, 28, 235), focus_color=(34, 34, 52, 245),
                 border_color=(90, 160, 255, 255), **kw):
        kw.setdefault("size", (240, 38))
        super().__init__(**kw)
        self.placeholder = placeholder
        self.password = password
        self.max_length = max_length
        self.on_change = on_change
        self.on_submit = on_submit
        self.color = color
        self.focus_color = focus_color
        self.border_color = border_color
        self.text = text[:max_length]
        self.cursor = len(self.text)
        self.sel_anchor = self.cursor  # other end of the selection (== cursor: none)
        self.focused = False
        self._blink_start = 0.0
        self._scroll = 0.0
        self._caret_x = 0.0
        self._sel_x = None  # (x0, x1) of the selection, relative to the text origin
        self.label = self.add(Label("", font_size=font_size, anchor=Anchor.MIDDLE_LEFT,
                                    offset=(self.PAD, 0), shadow=False))

    # ---- text ---------------------------------------------------------

    def set_text(self, text, notify=True):
        self.text = text[:self.max_length]
        self.cursor = self.sel_anchor = len(self.text)
        if notify:
            self._changed()

    def _changed(self):
        if self.on_change is not None:
            self.on_change(self.text)

    def _display(self):
        return "*" * len(self.text) if self.password else self.text

    def _prefix_width(self, n):
        font = self.manager.get_font(self.label.font, self.label._font_px())
        return font.size(self._display()[:n])[0] / self.manager.scale

    def _index_at(self, x):
        """Character index whose boundary is nearest logical x."""
        target = x - (self.rect[0] + self.PAD) + self._scroll
        best, best_d = 0, float("inf")
        for i in range(len(self.text) + 1):
            d = abs(self._prefix_width(i) - target)
            if d < best_d:
                best, best_d = i, d
        return best

    def _selection(self):
        """(lo, hi) of the selected range, or None."""
        if self.cursor == self.sel_anchor:
            return None
        return min(self.cursor, self.sel_anchor), max(self.cursor, self.sel_anchor)

    def _delete_selection(self):
        sel = self._selection()
        if sel is None:
            return False
        lo, hi = sel
        self.text = self.text[:lo] + self.text[hi:]
        self.cursor = self.sel_anchor = lo
        return True

    def _move(self, index, extend):
        self.cursor = index
        if not extend:
            self.sel_anchor = index

    # ---- layout / draw ---------------------------------------------------

    def _arrange_children(self):
        display = self._display()
        if self.manager is not None:
            caret = self._prefix_width(self.cursor)
            avail = max(1.0, self.rect[2] - 2 * self.PAD)
            if caret - self._scroll > avail:
                self._scroll = caret - avail
            if caret < self._scroll:
                self._scroll = caret
            self._scroll = max(0.0, self._scroll)
            self._caret_x = caret - self._scroll
            sel = self._selection()
            self._sel_x = ((self._prefix_width(sel[0]) - self._scroll,
                            self._prefix_width(sel[1]) - self._scroll) if sel else None)
        self.label.offset = (self.PAD - self._scroll, 0)
        if display or self.focused:
            self.label.text, self.label.color = display, _WHITE
        else:
            self.label.text, self.label.color = self.placeholder, (140, 140, 152, 255)
        super()._arrange_children()

    def draw(self, out):
        x, y, w, h = self.rect
        out.rect(self.rect, _color(self.focus_color if self.focused else self.color))
        if self.focused:
            b = _color(self.border_color)
            out.rect((x, y, w, 2), b)
            out.rect((x, y + h - 2, w, 2), b)
            out.rect((x, y, 2, h), b)
            out.rect((x + w - 2, y, 2, h), b)
        out.push_clip((x + 2, y + 2, w - 4, h - 4))
        if self._sel_x is not None:
            x0, x1 = self._sel_x
            out.rect((x + self.PAD + x0, y + h * 0.15, x1 - x0, h * 0.7), _color((70, 110, 190, 200)))
        super().draw(out)
        if self.focused and int((time.monotonic() - self._blink_start) * 2) % 2 == 0:
            out.rect((x + self.PAD + self._caret_x, y + h * 0.2, 2, h * 0.6), _color(_WHITE))
        out.pop_clip()

    # ---- input ----------------------------------------------------------

    def on_focus(self):
        self._blink_start = time.monotonic()

    def on_blur(self):
        pass

    def on_press(self, x, y):
        self.manager.set_focus(self)
        self._move(self._index_at(x), extend=False)
        self._blink_start = time.monotonic()

    def on_drag(self, x, y):
        self._move(self._index_at(x), extend=True)
        self._blink_start = time.monotonic()

    def on_release(self, x, y, inside):
        pass

    def on_key(self, event):
        k = event.key
        extend = bool(event.mod & pygame.KMOD_SHIFT)
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        t, c = self.text, self.cursor
        sel = self._selection()
        if k == pygame.K_BACKSPACE:
            if self._delete_selection():
                self._changed()
            elif c > 0:
                self.text, self.cursor = t[:c - 1] + t[c:], c - 1
                self.sel_anchor = self.cursor
                self._changed()
        elif k == pygame.K_DELETE:
            if self._delete_selection():
                self._changed()
            elif c < len(t):
                self.text = t[:c] + t[c + 1:]
                self._changed()
        elif k == pygame.K_LEFT:
            self._move(sel[0] if sel and not extend else max(0, c - 1), extend)
        elif k == pygame.K_RIGHT:
            self._move(sel[1] if sel and not extend else min(len(t), c + 1), extend)
        elif k == pygame.K_HOME:
            self._move(0, extend)
        elif k == pygame.K_END:
            self._move(len(t), extend)
        elif k == pygame.K_a and ctrl:
            self.sel_anchor, self.cursor = 0, len(t)
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.on_submit is not None:
                self.on_submit(self.text)
        elif k == pygame.K_ESCAPE:
            self.manager.set_focus(None)
        self._blink_start = time.monotonic()

    def on_text(self, text):
        text = "".join(ch for ch in text if ch.isprintable())
        if not text:
            return
        removed = self._delete_selection()
        text = text[:max(0, self.max_length - len(self.text))]
        if text:
            self.text = self.text[:self.cursor] + text + self.text[self.cursor:]
            self.cursor += len(text)
            self.sel_anchor = self.cursor
        if text or removed:
            self._blink_start = time.monotonic()
            self._changed()


class Dropdown(Widget):
    """A closed box showing the current option; click to open a list of
    all options below it, click one to select. on_change(index, option)
    fires when the selection changes."""

    interactive = True
    TEXT_PAD = 10.0

    def __init__(self, options, selected=0, on_change=None, font_size=22,
                 color=(30, 30, 40, 235), hover_color=(60, 60, 78, 245),
                 popup_color=(22, 22, 32, 252), highlight_color=(70, 110, 190, 255), **kw):
        kw.setdefault("size", (240, 38))
        super().__init__(**kw)
        self.options = list(options)
        self.selected = selected
        self.on_change = on_change
        self.color = color
        self.hover_color = hover_color
        self.popup_color = popup_color
        self.highlight_color = highlight_color
        self.open = False
        self._hover_index = -1
        self.label = self.add(Label(self.options[selected], font_size=font_size,
                                    anchor=Anchor.MIDDLE_LEFT, offset=(self.TEXT_PAD, 0)))
        self.add(Label("v", font_size=font_size, anchor=Anchor.MIDDLE_RIGHT, offset=(-12, 0)))
        # Not tree children (they only ever draw in the popup pass) - so
        # they need manager attachment and cleanup handled by hand below.
        self._option_labels = [Label(o, font_size=font_size, shadow=False) for o in self.options]

    @property
    def value(self):
        return self.options[self.selected]

    def set_selected(self, index, notify=True):
        if index == self.selected or not 0 <= index < len(self.options):
            return
        self.selected = index
        self.label.text = self.options[index]
        if notify and self.on_change is not None:
            self.on_change(index, self.options[index])

    def _attach(self, manager):
        super()._attach(manager)
        for lbl in self._option_labels:
            lbl.manager = manager

    def _release_resources(self):
        super()._release_resources()
        for lbl in self._option_labels:
            lbl._release_resources()

    # ---- popup ---------------------------------------------------------

    def _popup_rect(self):
        x, y, w, h = self.rect
        return x, y + h, w, h * len(self.options)

    def popup_contains(self, x, y):
        if not self.open:
            return False
        px, py, pw, ph = self._popup_rect()
        return px <= x < px + pw and py <= y < py + ph

    def close(self):
        self.open = False
        self._hover_index = -1
        if self.manager is not None and self.manager.popup is self:
            self.manager.popup = None

    def on_press(self, x, y):
        if self.open:
            if self.popup_contains(x, y):
                self.set_selected(int((y - self.rect[1] - self.rect[3]) // self.rect[3]))
            self.close()
        else:
            self.open = True
            self.manager.popup = self

    def on_release(self, x, y, inside):
        pass

    def on_hover(self, x, y):
        self._hover_index = (int((y - self.rect[1] - self.rect[3]) // self.rect[3])
                             if self.popup_contains(x, y) else -1)

    # ---- draw ----------------------------------------------------------

    def draw(self, out):
        out.rect(self.rect, _color(self.hover_color if (self.hovered or self.open) else self.color))
        super().draw(out)
        if self.open:
            out.defer(self._draw_popup)

    def _draw_popup(self, out):
        x, y, w, h = self.rect
        out.rect(self._popup_rect(), _color(self.popup_color))
        for i, lbl in enumerate(self._option_labels):
            row_y = y + h * (i + 1)
            if i == self._hover_index:
                out.rect((x, row_y, w, h), _color(self.highlight_color))
            lw, lh = lbl.measure(0, 0)
            lbl.arrange(x + self.TEXT_PAD, row_y + (h - lh) / 2, lw, lh)
            lbl.draw(out)
