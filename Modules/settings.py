"""
Player-facing graphics options, kept in ~/.ratwar/settings.json.

`settings` is one shared object: systems read its attributes when they need
them (the scene each frame, the death effect when someone dies), so a change
from the menu applies straight away.
"""

import json
import os

_PATH = os.path.join(os.path.expanduser("~"), ".ratwar", "settings.json")
_DEFAULTS = {"ssr": True, "gibs": True}


class Settings:
    def __init__(self):
        self.ssr = _DEFAULTS["ssr"]        # screen-space reflections (water)
        self.gibs = _DEFAULTS["gibs"]      # physical gibs on death (off = a cheap blood burst)
        self.load()

    def load(self):
        try:
            with open(_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for key, default in _DEFAULTS.items():
            setattr(self, key, bool(data.get(key, default)))

    def save(self):
        try:
            os.makedirs(os.path.dirname(_PATH), exist_ok=True)
            with open(_PATH, "w", encoding="utf-8") as f:
                json.dump({key: getattr(self, key) for key in _DEFAULTS}, f)
        except OSError:
            pass


settings = Settings()
