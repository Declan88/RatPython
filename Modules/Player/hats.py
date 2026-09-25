"""
The optional hats baked into rat.glb (the hidden "hat_*" nodes - see
skeletal_loader.py). A hat is identified everywhere (menu, network packet,
Scene.set_skeletal_hat) by its short name, e.g. "cowboy"; None means bare-headed.
"""

from Modules.Graphics.skeletal_loader import list_hat_names

RAT_MODEL_PATH = "Assets/Models/rat.glb"
_cache = None


def available_hats():
    """Short names of every hat in the rat model (read once)."""
    global _cache
    if _cache is None:
        _cache = list_hat_names(RAT_MODEL_PATH)
    return list(_cache)


def display_name(hat):
    return hat.replace("_", " ").title()
