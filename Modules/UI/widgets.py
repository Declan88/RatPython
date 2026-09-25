"""
Widget tree + layout for the UI system (see manager.py for the owner and
renderer.py for drawing).

Layout model (the same one Unity's RectTransform / Godot's Control use):
every widget is placed by
  - anchor: a point on its PARENT's content rect, as a fraction (0,0 is
    top-left, 1,1 is bottom-right) - see Anchor for named presets;
  - pivot:  a point on the widget ITSELF, same fraction convention, that
    gets pinned to the anchor point. Defaults to the anchor, so a widget
    anchored to a corner sits fully INSIDE that corner instead of
    hanging off it;
  - offset: extra (x, y) shift from there, y pointing down;
  - size + size_frac: final size = size + size_frac * parent size, so
    (0,0)+(1,1) fills the parent and (-40,-40)+(1,1) fills it with a
    20px margin all around.
All values are in "logical" pixels - UIManager multiplies by a resolution
scale (window height / reference height) when drawing, so a layout looks
the same on any window size.

A Panel with layout="vertical"/"horizontal" ignores its children's
anchors and stacks them instead (spacing, padding, cross-axis align,
optional fit_content), for menus and lists.
"""

import pygame


class Anchor:
    TOP_LEFT = (0.0, 0.0)
    TOP_CENTER = (0.5, 0.0)
    TOP_RIGHT = (1.0, 0.0)
    MIDDLE_LEFT = (0.0, 0.5)
    CENTER = (0.5, 0.5)
    MIDDLE_RIGHT = (1.0, 0.5)
    BOTTOM_LEFT = (0.0, 1.0)
    BOTTOM_CENTER = (0.5, 1.0)
    BOTTOM_RIGHT = (1.0, 1.0)


def _color(c):
    """(r, g, b[, a]) in 0-255 -> (r, g, b, a) floats in 0-1."""
    if len(c) == 3:
        c = (*c, 255)
    return tuple(v / 255.0 for v in c)


class Widget:
    interactive = False
    scrollable = False
    wheel_step = 48.0
    hovered = False
    pressed = False

    def __init__(self, anchor=Anchor.TOP_LEFT, pivot=None, offset=(0, 0),
                 size=(0, 0), size_frac=(0, 0), visible=True, name=None):
        self.anchor = tuple(anchor)
        self.pivot = tuple(pivot) if pivot is not None else self.anchor
        self.offset = tuple(offset)
        self.size = tuple(size)
        self.size_frac = tuple(size_frac)
        self.visible = visible
        self.name = name
        self.parent = None
        self.children = []
        self.manager = None
        self.rect = (0.0, 0.0, 0.0, 0.0)  # logical px: x, y, w, h - set by arrange()

    # ---- tree -------------------------------------------------------

    def add(self, child):
        child.parent = self
        self.children.append(child)
        child._attach(self.manager)
        return child

    def remove(self, child):
        self.children.remove(child)
        child.parent = None
        child._release_resources()
        child._attach(None)

    def clear_children(self):
        for c in list(self.children):
            self.remove(c)

    def _attach(self, manager):
        self.manager = manager
        for c in self.children:
            c._attach(manager)

    def _release_resources(self):
        for c in self.children:
            c._release_resources()

    # ---- layout -----------------------------------------------------

    def measure(self, parent_w, parent_h):
        return (
            max(0.0, self.size[0] + self.size_frac[0] * parent_w),
            max(0.0, self.size[1] + self.size_frac[1] * parent_h),
        )

    def arrange(self, x, y, w, h):
        self.rect = (x, y, w, h)
        self._arrange_children()

    def _content_rect(self):
        return self.rect

    def _arrange_children(self):
        cx, cy, cw, ch = self._content_rect()
        for c in self.children:
            if not c.visible:
                continue
            w, h = c.measure(cw, ch)
            x = cx + c.anchor[0] * cw + c.offset[0] - c.pivot[0] * w
            y = cy + c.anchor[1] * ch + c.offset[1] - c.pivot[1] * h
            c.arrange(x, y, w, h)

    # ---- draw / input ----------------------------------------------

    def draw(self, out):
        for c in self.children:
            if c.visible:
                c.draw(out)

    def pick(self, x, y):
        """Topmost visible interactive widget under logical point (x, y)."""
        for c in reversed(self.children):
            if not c.visible:
                continue
            hit = c.pick(x, y)
            if hit is not None:
                return hit
        if self.interactive and self._contains(x, y):
            return self
        return None

    def _contains(self, x, y):
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def pick_scroll(self, x, y):
        """Topmost visible scrollable widget under logical point (x, y)."""
        for c in reversed(self.children):
            if c.visible:
                hit = c.pick_scroll(x, y)
                if hit is not None:
                    return hit
        return None

    # Mouse capture protocol, driven by UIManager.handle_event for an
    # interactive widget (see its docstring). Coordinates are logical px.
    def on_press(self, x, y):
        pass

    def on_drag(self, x, y):
        pass

    def on_hover(self, x, y):
        """Cursor moved while over this widget (and nothing is held)."""

    def on_release(self, x, y, inside):
        if inside:
            self.click()

    def click(self):
        pass

    def scroll_by(self, dy):
        pass


class Panel(Widget):
    def __init__(self, color=(0, 0, 0, 0), layout=None, spacing=0, padding=0,
                 align="start", fit_content=False, **kw):
        super().__init__(**kw)
        self.color = color
        self.layout = layout
        self.spacing = spacing
        self.padding = padding
        self.align = align
        self.fit_content = fit_content

    def _content_rect(self):
        x, y, w, h = self.rect
        p = self.padding
        return (x + p, y + p, max(0.0, w - 2 * p), max(0.0, h - 2 * p))

    def measure(self, parent_w, parent_h):
        w, h = super().measure(parent_w, parent_h)
        if self.fit_content and self.layout:
            kids = [c.measure(w, h) for c in self.children if c.visible]
            if kids:
                vertical = self.layout == "vertical"
                gaps = self.spacing * (len(kids) - 1)
                main = sum(k[1] if vertical else k[0] for k in kids) + gaps
                cross = max(k[0] if vertical else k[1] for k in kids)
                cw, ch = (cross, main) if vertical else (main, cross)
                w = max(w, cw + 2 * self.padding)
                h = max(h, ch + 2 * self.padding)
        return w, h

    def _arrange_children(self):
        if not self.layout:
            return super()._arrange_children()
        cx, cy, cw, ch = self._content_rect()
        vertical = self.layout == "vertical"
        kids = [c for c in self.children if c.visible]
        pos = 0.0
        for c in kids:
            w, h = c.measure(cw, ch)
            main, cross = (h, w) if vertical else (w, h)
            avail = cw if vertical else ch
            shift = {"start": 0.0, "center": (avail - cross) / 2, "end": avail - cross}[self.align]
            if vertical:
                x, y = cx + shift + c.offset[0], cy + pos + c.offset[1]
            else:
                x, y = cx + pos + c.offset[0], cy + shift + c.offset[1]
            c.arrange(x, y, w, h)
            pos += main + self.spacing

    def draw(self, out):
        if self.color[3] > 0:
            out.rect(self.rect, _color(self.color))
        super().draw(out)


class Label(Widget):
    """Text. With size == (0, 0) (the default) it sizes itself to the text.
    The text texture is rendered white and tinted by `color` when drawn,
    so recoloring (or fading via alpha) never re-renders anything - only
    changing text/font_size/font/shadow or the resolution scale does."""

    def __init__(self, text="", font_size=24, color=(255, 255, 255, 255),
                 align="left", shadow=True, font=None, **kw):
        super().__init__(**kw)
        self.text = text
        self.font_size = font_size
        self.color = color
        self.align = align
        self.shadow = shadow
        self.font = font
        self._tex = None
        self._tex_key = None
        self._tex_size = (0, 0)

    def _font_px(self):
        return max(1, round(self.font_size * self.manager.scale))

    def measure(self, parent_w, parent_h):
        if self.size == (0, 0) and self.size_frac == (0, 0) and self.manager is not None:
            font = self.manager.get_font(self.font, self._font_px())
            lines = self.text.split("\n")
            w = max(font.size(line or " ")[0] for line in lines)
            h = font.get_linesize() * len(lines)
            s = self.manager.scale
            return (w / s, h / s)
        return super().measure(parent_w, parent_h)

    def _ensure_texture(self):
        px = self._font_px()
        key = (self.text, px, self.font, self.shadow, self.align)
        if key == self._tex_key:
            return
        self._release_resources()
        font = self.manager.get_font(self.font, px)
        lines = [line or " " for line in self.text.split("\n")]
        rendered = [font.render(line, True, (255, 255, 255)) for line in lines]
        shadow = [font.render(line, True, (0, 0, 0)) for line in lines] if self.shadow else None
        so = max(1, px // 16) if self.shadow else 0
        line_h = font.get_linesize()
        width = max(r.get_width() for r in rendered)
        surf = pygame.Surface((width + so, line_h * len(lines) + so), pygame.SRCALPHA)
        for i, r in enumerate(rendered):
            ax = {"left": 0, "center": (width - r.get_width()) // 2,
                  "right": width - r.get_width()}[self.align]
            if shadow:
                surf.blit(shadow[i], (ax + so, i * line_h + so))
            surf.blit(r, (ax, i * line_h))
        self._tex = self.manager.renderer.texture_from_surface(surf)
        self._tex_key = key
        self._tex_size = surf.get_size()

    def _release_resources(self):
        super()._release_resources()
        if self._tex is not None:
            self._tex.release()
            self._tex = None
            self._tex_key = None

    def draw(self, out):
        if not self.text:
            return
        self._ensure_texture()
        tw, th = self._tex_size
        x, y, w, h = self.rect
        s = self.manager.scale
        # Aligned inside the widget's own rect (matters when it's wider
        # than its text), drawn at the texture's exact pixel size so the
        # glyphs stay crisp rather than being resampled.
        px = x * s + {"left": 0, "center": (w * s - tw) / 2, "right": w * s - tw}[self.align]
        out.texture(self._tex, px, y * s, tw, th, _color(self.color))


class Image(Widget):
    """A texture from `path` (any format pygame loads), drawn stretched to
    the widget's size - or at its natural pixel size if size is (0, 0)."""

    def __init__(self, path=None, tint=(255, 255, 255, 255), **kw):
        super().__init__(**kw)
        self.path = path
        self.tint = tint
        self._tex = None
        self._natural = (0, 0)

    def measure(self, parent_w, parent_h):
        if self.size == (0, 0) and self.size_frac == (0, 0):
            self._load()
            return self._natural
        return super().measure(parent_w, parent_h)

    def _load(self):
        if self._tex is None and self.path and self.manager is not None:
            surf = pygame.image.load(self.path).convert_alpha()
            self._tex = self.manager.renderer.texture_from_surface(surf)
            self._natural = surf.get_size()

    def _release_resources(self):
        super()._release_resources()
        if self._tex is not None:
            self._tex.release()
            self._tex = None

    def draw(self, out):
        self._load()
        if self._tex is not None:
            out.texture_logical(self._tex, self.rect, _color(self.tint))


class Button(Panel):
    interactive = True

    def __init__(self, text="Button", on_click=None, font_size=24,
                 color=(50, 50, 60, 220), hover_color=(80, 80, 100, 235),
                 pressed_color=(30, 30, 40, 240), text_color=(255, 255, 255, 255),
                 **kw):
        kw.setdefault("size", (160, 40))
        super().__init__(color=color, **kw)
        self.on_click = on_click
        self.normal_color = color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.hovered = False
        self.pressed = False
        self.label = self.add(Label(text, font_size=font_size, color=text_color,
                                    anchor=Anchor.CENTER))

    def draw(self, out):
        self.color = (self.pressed_color if self.pressed
                      else self.hover_color if self.hovered else self.normal_color)
        super().draw(out)

    def click(self):
        if self.on_click is not None:
            self.on_click()
