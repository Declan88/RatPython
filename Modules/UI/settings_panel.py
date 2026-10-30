"""
The player options (Modules/settings.py) as UI controls - shared by the main menu's settings card and the pause menu,
so both always offer the same options. Changes apply immediately and are saved.
"""

from . import theme
from .controls import Checkbox, Slider
from .widgets import Label


def add_settings_options(card):
    """Adds the graphics, HUD and controller options to `card` (a vertical Panel)."""
    card.add(Label("SETTINGS", font_size=28, color=theme.ACCENT))
    _graphics(card)
    card.add(Label("HUD", font_size=28, color=theme.ACCENT))
    _option(card, "Damage numbers", "damage_numbers")
    _option(card, "Combine hit + explosion damage", "combine_explosion_damage")
    card.add(Label("CONTROLLER", font_size=28, color=theme.ACCENT))
    _controller(card)


def _option(card, text, name):
    from Modules.settings import settings

    def changed(checked):
        setattr(settings, name, checked)
        settings.save()
    card.add(Checkbox(text, checked=getattr(settings, name), on_toggle=changed, font_size=26, size=(380, 40)))


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
