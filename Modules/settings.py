"""
Player-facing options, kept in ~/.ratwar/settings.json.

`settings` is one shared object: systems read its attributes when they need
them (the scene each frame, the death effect when someone dies), so a change
from the menu applies straight away.
"""

import json
import os

_PATH = os.path.join(os.path.expanduser("~"), ".ratwar", "settings.json")
_DEFAULTS = {
    "ssr": True,                      # screen-space reflections (water)
    "gibs": True,                     # physical gibs on death (off = a cheap blood burst)
    "bloom": True,                    # glow on "bloom_" objects (see Modules/Graphics/bloom.py)
    "damage_numbers": True,           # the damage our hits do, popping up on the target (Modules/UI/damage_numbers.py)
    "combine_explosion_damage": True, # one number for a direct hit plus its explosion's splash on the same player
    "controller_sensitivity": 1.0,    # right-stick look speed multiplier
    "controller_invert_y": False,
    "aim_assist": True,               # controller only (Modules/Input/aim_assist.py)
    "controller_vibration": True,
}


class Settings:
    def __init__(self):
        for key, default in _DEFAULTS.items():
            setattr(self, key, default)
        self.load()

    def load(self):
        try:
            with open(_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for key, default in _DEFAULTS.items():
            if key in data:
                try:
                    setattr(self, key, type(default)(data[key]))   # same type as the default (bool, float)
                except (TypeError, ValueError):
                    pass

    def save(self):
        try:
            os.makedirs(os.path.dirname(_PATH), exist_ok=True)
            with open(_PATH, "w", encoding="utf-8") as f:
                json.dump({key: getattr(self, key) for key in _DEFAULTS}, f)
        except OSError:
            pass


settings = Settings()
