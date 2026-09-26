"""
Blood for gibs and deaths, built in code as particle systems (the same kind a
.pcf holds - see particle_system.py) and registered on a ParticleManager:

  gore_blood_spurt  - a heartbeat-pulsed spray of streaks, for a gib to pump out
                      of its wound while it tumbles (spawn it with follow=gib's
                      position and follow_particles=False, inherit_velocity=0.6
                      so the blood trails behind the flying chunk)
  gore_blood_burst  - the fast spray of streaks thrown out when someone dies
  gore_blood_cloud  - a few soft red puffs hanging where they died

The streaks use the blood_mist sprite sheet, the puffs the smoke sprite (alpha
blended and tinted red; colours past 255 brighten the dark blood texture).
"""

import math

from .pcf import Function, ParticleSystemDef

_STREAK_MATERIAL = "particle/blood_mist/blood_mist.vmt"
_PUFF_MATERIAL = "particle/particle_smokegrenade.vmt"


def _emit_spurt(fx, params, state, t, dt):
    """A pulsing, fading emitter: rate = peak * envelope * heartbeat, for
    `duration` seconds (params: duration, peak_rate, beat_hz, decay)."""
    duration = float(params.get("duration", 2.0))
    if t >= duration:
        state["done"] = True
        return
    envelope = (1.0 - t / duration) ** float(params.get("decay", 1.5))
    beat = max(0.0, math.sin(2.0 * math.pi * float(params.get("beat_hz", 3.0)) * t)) ** 2
    sources = len(fx.anchor_points) if fx.anchor_points is not None else 1     # per-anchor rate
    state["emitted"] += float(params.get("peak_rate", 100.0)) * sources * envelope * (0.12 + 0.88 * beat) * dt
    whole = int(state["emitted"])
    state["emitted"] -= whole
    fx.emit(whole)


def _fn(name, **params):
    return Function(name, params)


def _streaks(name, count, speed, life, length, radius, gravity=-700.0, emitter=None):
    return ParticleSystemDef(
        name,
        {"max_particles": count if emitter is None else 1200, "material": _STREAK_MATERIAL, "radius": 1.0,
         "color": (255, 255, 255, 255)},
        emitters=[emitter or _fn("emit_instantaneously", num_to_emit=count)],
        initializers=[
            _fn("Position Within Sphere Random", distance_min=0.0, distance_max=1.5,
                speed_min=speed[0], speed_max=speed[1], speed_random_exponent=0.8),
            _fn("Velocity Random", speed_in_local_coordinate_system_min=(0.0, 0.0, speed[0] * 0.3),
                speed_in_local_coordinate_system_max=(0.0, 0.0, speed[1] * 0.5)),
            _fn("Lifetime Random", lifetime_min=life[0], lifetime_max=life[1]),
            _fn("Radius Random", radius_min=radius[0], radius_max=radius[1]),
            _fn("Sequence Random", sequence_min=0, sequence_max=31),
            _fn("Trail Length Random", length_min=length[0], length_max=length[1]),
            _fn("Color Random", color1=(700, 110, 95, 255), color2=(980, 170, 140, 255)),
            _fn("Alpha Random", alpha_min=210, alpha_max=255),
        ],
        operators=[
            _fn("Movement Basic", gravity=(0.0, 0.0, gravity), drag=0.015),
            _fn("Alpha Fade Out Random", **{"fade out time min": 0.25, "fade out time max": 0.45,
                                            "proportional 0/1": True}),
        ],
        renderers=[_fn("render_sprite_trail", **{"min length": 1.0, "max length": 45.0,
                                                 "length fade in time": 0.0, "animation rate": 0.0})],
    )


def _cloud():
    return ParticleSystemDef(
        "gore_blood_cloud",
        {"max_particles": 12, "material": _PUFF_MATERIAL, "radius": 12.0, "color": (255, 255, 255, 255)},
        emitters=[_fn("emit_instantaneously", num_to_emit=9)],
        initializers=[
            _fn("Position Within Sphere Random", distance_min=0.0, distance_max=6.0,
                speed_min=15.0, speed_max=70.0),
            _fn("Lifetime Random", lifetime_min=0.5, lifetime_max=1.0),
            _fn("Radius Random", radius_min=9.0, radius_max=16.0),
            _fn("Rotation Random", rotation_offset_min=0.0, rotation_offset_max=360.0),
            _fn("Alpha Random", alpha_min=110, alpha_max=170),
            _fn("Color Random", color1=(190, 8, 8, 255), color2=(110, 0, 0, 255)),
        ],
        operators=[
            _fn("Movement Basic", gravity=(0.0, 0.0, -80.0), drag=0.08),
            _fn("Radius Scale", start_time=0.0, end_time=1.0, radius_start_scale=0.4, radius_end_scale=1.6),
            _fn("Alpha Fade Out Random", **{"fade out time min": 0.6, "fade out time max": 0.8,
                                            "proportional 0/1": True}),
        ],
        renderers=[_fn("render_animated_sprites")],
    )


def register_blood(manager):
    """Adds the blood effects (and the pulsing emitter they use) to a ParticleManager."""
    manager.emitters["emit_spurt"] = _emit_spurt
    for definition in (
        _streaks("gore_blood_spurt", 0, speed=(30.0, 150.0), life=(0.35, 0.8), length=(0.02, 0.05),
                 radius=(0.5, 1.1),
                 emitter=_fn("emit_spurt", duration=1.8, peak_rate=140.0, beat_hz=3.2, decay=1.4)),
        _streaks("gore_blood_burst", 90, speed=(80.0, 380.0), life=(0.4, 1.0), length=(0.03, 0.08),
                 radius=(0.6, 1.4)),
        _cloud(),
    ):
        manager.definitions[definition.name] = definition
