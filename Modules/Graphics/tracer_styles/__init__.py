"""
Tracer styles: every .py file in this folder (except base.py) is imported at
startup and its STYLE (or STYLES list) registered by name. See base.py for how to
write one; weapons pick one with `tracer_style = "<name>"`.

The built-in styles are also imported by name below, because a packaged (Nuitka)
build only contains modules it can see imported - scanning the folder finds nothing
there. A style added as a new file is picked up by the scan when running from source;
for a packaged build, import it below too (or build with
--include-package=Modules.Graphics.tracer_styles).
"""

import importlib
import pkgutil

from .base import TracerStyle
from . import default as _default, laser as _laser

STYLES = {}


def register_style(style):
    STYLES[style.name] = style


def get_style(name):
    """The style called `name`, or the default if it's unknown/None."""
    return STYLES.get(name, STYLES["default"])


def _discover():
    for module in (_default, _laser):
        register_style(module.STYLE)
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name == "base" or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{module_info.name}")
        found = list(getattr(module, "STYLES", ()))
        if getattr(module, "STYLE", None) is not None:
            found.append(module.STYLE)
        for style in found:
            register_style(style)


_discover()
