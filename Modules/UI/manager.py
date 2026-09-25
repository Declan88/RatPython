"""
UIManager: owns the widget tree root, the resolution scale, the font
cache, mouse input, and the per-frame layout + draw.

Usage (see app.py / demo.py):
    ui = UIManager(window)
    ui.root.add(Label("HP 100", anchor=Anchor.BOTTOM_LEFT, offset=(24, -24)))
    ...
    ui.render()            # once per frame, after the scene, before flip
    window.handle_events(..., event_filter=ui.handle_event)

Costs nothing while every root child is hidden (render() returns before
layout or any GL work), so a HUD/menu can be built up front and toggled.

The game normally grabs and hides the mouse for camera look. Interactive
widgets (Button) only make sense with a free cursor: ui.set_cursor_free(
True) shows/releases it, and while it's free handle_event() swallows mouse
events so the camera doesn't spin under the menu.
"""

import pygame

from . import theme
from .renderer import UIRenderer
from .widgets import Widget


class SmoothFont:
    """A pygame Font that rasterizes at SUPERSAMPLE x the requested size and
    scales the result down. pygame's own glyph rendering hints/rounds every
    glyph to whole pixels, which at larger sizes shows as uneven letter
    spacing and jagged curves; averaging a larger render gives even spacing
    and smooth edges. Same size()/get_linesize()/render() surface as Font, so
    callers are unchanged."""

    SUPERSAMPLE = 3
    # An OS typeface runs bigger than pygame's bundled one at the same px, so
    # it's scaled down to keep the layout's sizes meaning what they always did.
    OS_FONT_SCALE = 0.8

    def __init__(self, path, px):
        if path is not None:
            px = max(1, round(px * self.OS_FONT_SCALE))
        self._big = pygame.font.Font(path, px * self.SUPERSAMPLE)

    def _down(self, n):
        return -(-n // self.SUPERSAMPLE)

    def size(self, text):
        w, h = self._big.size(text)
        return self._down(w), self._down(h)

    def get_linesize(self):
        return self._down(self._big.get_linesize())

    def render(self, text, antialias, color):
        big = self._big.render(text, True, color)
        w, h = big.get_size()
        return pygame.transform.smoothscale(big, (max(1, self._down(w)), max(1, self._down(h))))


class UIManager:
    KEY_REPEAT_DELAY_MS = 400
    KEY_REPEAT_INTERVAL_MS = 35

    def __init__(self, window, reference_height=1080, scale_mode="height"):
        """scale_mode "height": logical units scale with window height
        relative to reference_height (a 24px label is 24px on a 1080p
        window, 12px on 540p); "fixed": 1 logical unit = 1 pixel."""
        self.window = window
        self.reference_height = reference_height
        self.scale_mode = scale_mode
        self.scale = 1.0
        self.root = Widget(name="root")
        self.root.manager = self
        self.renderer = None  # created on first render - layout/tests need no GL
        self.cursor_free = False
        self._fonts = {}
        self._hover = None
        self._pressed = None
        self._mouse = None
        self.popup = None  # open Dropdown, if any (see controls.Dropdown)
        self.focus = None  # focused TextInput, if any

    # ---- resources --------------------------------------------------

    def get_font(self, path, px):
        path = path or theme.font_path()
        key = (path, px)
        font = self._fonts.get(key)
        if font is None:
            if not pygame.font.get_init():
                pygame.font.init()
            font = self._fonts[key] = SmoothFont(path, px)
        return font

    def _ensure_renderer(self):
        if self.renderer is None:
            self.renderer = UIRenderer(self.window.ctx)

    # ---- layout / draw ---------------------------------------------

    def layout(self):
        w, h = self.window.width, self.window.height
        self.scale = h / self.reference_height if self.scale_mode == "height" else 1.0
        self.root.arrange(0.0, 0.0, w / self.scale, h / self.scale)

    def render(self):
        if not any(c.visible for c in self.root.children):
            return
        self._ensure_renderer()
        self.layout()
        draw_list = self.renderer.new_draw_list(self.scale)
        self.root.draw(draw_list)
        while draw_list.overlays:
            draw_list.overlays.pop(0)(draw_list)
        self.renderer.draw(draw_list, self.window.width, self.window.height)

    # ---- input ------------------------------------------------------

    def set_cursor_free(self, free):
        if free == self.cursor_free:
            return
        self.cursor_free = free
        pygame.mouse.set_visible(free)
        pygame.event.set_grab(not free)
        if free:
            pygame.mouse.set_pos((self.window.width // 2, self.window.height // 2))
        else:
            self._set_hover(None)
            self.set_focus(None)
            self._close_popup()
            pygame.event.clear(pygame.MOUSEMOTION)
            pygame.mouse.get_rel()

    def _pick(self, x, y):
        # An open dropdown's list floats above everything, including
        # widgets laid out (and hit-tested) after it.
        if self.popup is not None and self.popup.popup_contains(x, y):
            return self.popup
        return self.root.pick(x, y)

    def _close_popup(self):
        if self.popup is not None:
            self.popup.close()

    def set_focus(self, widget):
        """Give keyboard focus to a TextInput (or None). While something is
        focused, handle_event swallows all key/text events (ESC included -
        it blurs instead of quitting the game)."""
        if widget is self.focus:
            return
        if self.focus is not None:
            self.focus.focused = False
            self.focus.on_blur()
        self.focus = widget
        if widget is not None:
            widget.focused = True
            widget.on_focus()
            pygame.key.start_text_input()
            # OS-style repeat (hold Backspace/arrows) only while typing -
            # the game's own KEYDOWN handling shouldn't see repeats.
            pygame.key.set_repeat(self.KEY_REPEAT_DELAY_MS, self.KEY_REPEAT_INTERVAL_MS)
        else:
            pygame.key.stop_text_input()
            pygame.key.set_repeat()

    def _logical(self, pos):
        return pos[0] / self.scale, pos[1] / self.scale

    def _set_hover(self, target):
        if target is self._hover:
            return
        if self._hover is not None:
            self._hover.hovered = False
        self._hover = target
        if target is not None:
            target.hovered = True

    def handle_event(self, event):
        """Returns True if the event was consumed (WindowManager then skips
        its own handling, e.g. mouse look). Only ever consumes mouse events,
        and only while the cursor is free.

        A left press on an interactive widget captures the mouse for it:
        on_press, then on_drag for every motion event (even outside the
        widget - a slider keeps following a drag off its own edge), then
        on_release with whether the cursor ended up over it."""
        if not self.cursor_free:
            return False
        t = event.type
        if self.focus is not None and t in (pygame.KEYDOWN, pygame.KEYUP, pygame.TEXTINPUT,
                                            pygame.TEXTEDITING):
            if t == pygame.KEYDOWN:
                self.focus.on_key(event)
            elif t == pygame.TEXTINPUT:
                self.focus.on_text(event.text)
            return True
        if t == pygame.MOUSEMOTION:
            self._mouse = event.pos
            x, y = self._logical(event.pos)
            if self._pressed is not None:
                self._pressed.on_drag(x, y)
            else:
                hover = self._pick(x, y)
                self._set_hover(hover)
                if hover is not None:
                    hover.on_hover(x, y)
            return True
        if t == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._mouse = event.pos
            x, y = self._logical(event.pos)
            target = self._pick(x, y)
            if self.popup is not None and target is not self.popup:
                self._close_popup()
            if target is not self.focus:
                self.set_focus(None)
            if target is not None:
                target.pressed = True
                self._pressed = target
                target.on_press(x, y)
            return True
        if t == pygame.MOUSEBUTTONUP and event.button == 1:
            self._mouse = event.pos
            x, y = self._logical(event.pos)
            pressed, self._pressed = self._pressed, None
            hit = self._pick(x, y)
            if pressed is not None:
                pressed.pressed = False
                pressed.on_release(x, y, hit is pressed)
            self._set_hover(hit)
            return True
        if t == pygame.MOUSEWHEEL:
            x, y = self._logical(self._mouse or pygame.mouse.get_pos())
            target = self.root.pick_scroll(x, y)
            if target is not None:
                target.scroll_by(-event.y * target.wheel_step)
            return True
        return t in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP)

    # ---- teardown ---------------------------------------------------

    def destroy(self):
        self.root._release_resources()
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
