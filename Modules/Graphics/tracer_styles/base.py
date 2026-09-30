"""
Base class for a tracer look. To add a new tracer, drop a .py file in this folder
that subclasses TracerStyle and exposes an instance called STYLE (or a list called
STYLES) - it's discovered automatically (see __init__.py) - then point a weapon at
it with `tracer_style = "<name>"`. See default.py (a moving streak) and laser.py
(a static noisy beam that fades out) for two complete examples.

A style is two things:

  * segment(age, distance): the TIMING - which part of the path (start -> end, in
    metres) is drawn `age` seconds after the shot, or None once it's over.
  * fragment: the LOOK - a GLSL function defining
        vec3 shade(vec2 uv, float fade, float metres, float total, float seed)
    that returns the colour (drawn ADDITIVELY, so black = invisible).
        uv.x    0 at the ribbon's tail -> 1 at its head
        uv.y    -1..1 across the ribbon
        fade    the intensity segment() returned (also thins the ribbon a bit)
        metres  distance along the path from the muzzle to this pixel
        total   the whole path's length in metres
        seed    a random number per shot, so noise differs shot to shot
"""


class TracerStyle:
    name = "unnamed"
    width = 1.0          # multiplier on the base ribbon width
    shrink = 0.65        # how much thinner the ribbon gets as `fade` drops to 0 (0 = constant width)
    fragment = ""        # GLSL defining shade() - see the module docstring

    def segment(self, age, distance):
        """Returns None when the tracer is finished, else
        (a, b, u0, u1, fade): draw the path from `a` to `b` metres, with the
        ribbon's uv.x running u0 -> u1 across it, at intensity `fade` (0..1)."""
        raise NotImplementedError
