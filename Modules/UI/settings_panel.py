"""
The player options (Modules/settings.py) as UI controls - shared by the main menu's settings card and the pause menu,
so both always offer the same options. Changes apply immediately and are saved. They're split into categories, one
tab each (see _CATEGORIES).
"""

from . import theme
from .controls import Checkbox, Slider
from .widgets import Button, Label, Panel


def add_settings_options(card):
    """Adds the options to `card` (a vertical Panel): a row of category tabs, and under it the chosen category's
    options (the first one's to begin with)."""
    card.add(Label("SETTINGS", font_size=28, color=theme.ACCENT))
    tabs = card.add(Panel(layout="horizontal", spacing=8, fit_content=True))
    pages, buttons = [], []

    def show(index):
        for i, (page, button) in enumerate(zip(pages, buttons)):
            page.visible = i == index
            button.normal_color = theme.ACCENT_DIM if i == index else theme.BUTTON["color"]

    for i, (title, build) in enumerate(_CATEGORIES):
        buttons.append(tabs.add(Button(title, on_click=lambda i=i: show(i), size=(124, 44), font_size=22,
                                       **theme.BUTTON)))
        page = card.add(Panel(layout="vertical", spacing=getattr(card, "spacing", 12), fit_content=True))
        build(page)
        pages.append(page)
    show(0)


def _video(card):
    _frame_rate(card)
    _graphics(card)


def _hud(card):
    _option(card, "Damage numbers", "damage_numbers")
    _option(card, "Combine hit + explosion damage", "combine_explosion_damage")


def _option(card, text, name):
    from Modules.settings import settings

    def changed(checked):
        setattr(settings, name, checked)
        settings.save()
    card.add(Checkbox(text, checked=getattr(settings, name), on_toggle=changed, font_size=26, size=(380, 40)))


_FPS_MIN, _FPS_MAX, _FPS_STEP = 30, 300, 10
_FPS_UNCAPPED_STOP = _FPS_MAX + _FPS_STEP     # the slider's last stop, past 300: no limit


def _frame_rate(card):
    """The frame rate: a slider from 30 to 300 (its last stop: uncapped) and a "match the monitor" box - the
    default, the monitor's refresh rate. The game runs up to FPS_HEADROOM above whichever is chosen."""
    from Modules.settings import settings
    from Modules.Window.window import monitor_refresh_rate

    controls = {}

    def slider_changed(position):
        settings.fps_limit = -1 if position >= _FPS_UNCAPPED_STOP else int(position)
        controls["match"].checked = False
        settings.save()

    def match_toggled(checked):
        # Ticking it goes back to the monitor's rate; unticking keeps that rate, now as a fixed choice.
        rate = min(_FPS_MAX, max(_FPS_MIN, monitor_refresh_rate()))
        settings.fps_limit = 0 if checked else rate
        controls["slider"].set_value(rate, notify=False)
        settings.save()

    value = settings.fps_limit
    start = _FPS_UNCAPPED_STOP if value < 0 else (value or min(_FPS_MAX, max(_FPS_MIN, monitor_refresh_rate())))
    controls["slider"] = card.add(Slider(value=start, min_value=_FPS_MIN, max_value=_FPS_UNCAPPED_STOP,
                                         step=_FPS_STEP, on_change=slider_changed, size=(380, 32)))
    controls["match"] = card.add(Checkbox("Match monitor refresh rate", checked=value == 0, on_toggle=match_toggled,
                                          font_size=26, size=(380, 40)))


def _graphics(card):
    _option(card, "Reflections (SSR)", "ssr")
    _option(card, "Gibs on death", "gibs")
    _option(card, "Bloom", "bloom")
    card.add(Label("Off: water uses a cheap reflection,", font_size=20, color=theme.MUTED))
    card.add(Label("deaths are a blood burst, not gibs,", font_size=20, color=theme.MUTED))
    card.add(Label("and lights don't glow.", font_size=20, color=theme.MUTED))


def _controller(card):
    from Modules.settings import settings
    label = card.add(Label("", font_size=24))

    def sensitivity_changed(value):
        settings.controller_sensitivity = round(value, 2)
        label.text = f"Look sensitivity: {settings.controller_sensitivity:.2f}x"
        settings.save()

    sensitivity_changed(settings.controller_sensitivity)
    card.add(Slider(value=settings.controller_sensitivity, min_value=0.3, max_value=2.5, step=0.05,
                    on_change=sensitivity_changed, size=(380, 32)))
    _option(card, "Invert look up/down", "controller_invert_y")
    _option(card, "Aim assist", "aim_assist")
    _option(card, "Vibration", "controller_vibration")


# The settings' categories, one tab each, in order: (tab title, what fills its page).
_CATEGORIES = (
    ("VIDEO", _video),
    ("HUD", _hud),
    ("CONTROLLER", _controller),
)
