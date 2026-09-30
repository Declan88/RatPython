"""
A lightning strike for the /smite command, built in code as particle systems
(the same kind a .pcf holds - see particle_system.py) and registered on a
ParticleManager - same shape as blood.py's own gore_* systems:

  smite_flash    - an instant, bright white-blue glow at the strike point
  smite_bolt     - the lower ~18m of a tight column of thick, fast-falling energy trails
                   (lgtning.png - see _BOLT_MATERIAL's own comment), dense enough to
                   reliably reach all the way down to the target's own feet every strike
                   (see _bolt_segment's own docstring on why this is split from
                   smite_bolt_sky rather than one system spanning the whole height) -
                   the particle vocabulary has no notion of a literal jagged bolt SHAPE
                   (see particle_system.py's own initializers), so a bundle of thick
                   trails plunging down a narrow column in near-zero time is the same
                   kind of abstraction blood.py's streaks already use for "blood spray":
                   it READS as one massive bolt slamming down from the sky, not a
                   physically exact one.
  smite_bolt_sky - the rest of the same column, ~18-90m up - sparser, since nobody's
                   looking closely that high up; purely for the "reaches toward the sky"
                   scale smite_bolt's own dense ground-level coverage doesn't need.
  smite_crackle  - a brief burst of thinner arcs radiating out from the
                   strike point once the bolt lands, for a crackling-
                   electricity impression at the point of impact.
"""

from .pcf import Function, ParticleSystemDef

# Was "particle/bendibeam.vmt" - that texture bakes its own green tint straight into the
# file (confirmed by looking at it), which dominates over Color Random's blue-white values
# below when multiplied, making the bolt/crackle read as green no matter what colour they're
# tinted. lgtning.png (Modules/Particles/dissolve_sparks.py's own from-scratch recreation of
# Source's sprites/lgtning.vmt look - see that module's docstring) was authored deliberately
# grayscale/white for exactly this reason, so it's reused here instead.
_BOLT_MATERIAL = "particle/lgtning.vmt"
_FLASH_MATERIAL = "particle/particle_glow_04.vmt"


def _fn(name, **params):
    return Function(name, params)


def _flash():
    return ParticleSystemDef(
        "smite_flash",
        {"max_particles": 3, "material": _FLASH_MATERIAL, "radius": 60.0, "color": (255, 255, 255, 255)},
        emitters=[_fn("emit_instantaneously", num_to_emit=3)],
        initializers=[
            _fn("Position Within Sphere Random", distance_min=0.0, distance_max=0.4, speed_min=0.0, speed_max=0.0),
            _fn("Lifetime Random", lifetime_min=0.12, lifetime_max=0.2),
            # Real ~0.4-0.66m base radius (÷0.0254 - see smite_bolt's own docstring on this
            # system-wide Source-units convention), not the original 1.14-1.78m: this project's
            # characters (rat.glb, _BODY_HEIGHT=1.39m) are much shorter than the human-scale
            # target this was originally tuned for, so a flash comparable to a PERSON'S height
            # swallows the whole rat from below-ground up past its head - reading as "the
            # strike is at the head" even though it's correctly centered at the feet.
            _fn("Radius Random", radius_min=16.0, radius_max=26.0),
            # Cool white with a faint blue edge - Color Random's own (r, g, b) 0-255 range,
            # same mechanism blood.py's own Color Random tints its grayscale-ish streaks with.
            _fn("Color Random", color1=(210, 225, 255, 255), color2=(255, 255, 255, 255)),
            _fn("Alpha Random", alpha_min=230, alpha_max=255),
        ],
        operators=[
            _fn("Radius Scale", start_time=0.0, end_time=1.0, radius_start_scale=0.3, radius_end_scale=1.4),
            _fn("Alpha Fade Out Random", **{"fade out time min": 0.03, "fade out time max": 0.08,
                                            "proportional 0/1": True}),
        ],
        renderers=[_fn("render_animated_sprites")],
    )


def _bolt_segment(name, max_particles, num_to_emit, z_min, z_max):
    """One vertical SEGMENT of the descending bolt column - shared by both smite_bolt (the
    dense lower trunk) and smite_bolt_sky (the sparse upper reach). Splitting the column
    into two systems instead of one spanning the whole ~90m height uniformly fixes a real
    problem: each particle only travels a few metres total (speed * its own short
    lifetime - see below) - it does NOT fall the whole height from where it starts, so
    whether the bolt visibly reaches the ground at all depends entirely on how many
    particles happened to spawn near the bottom. A single system spreading its particle
    budget UNIFORMLY across 90m left only ~2 particles landing anywhere in the bottom 2m
    out of 85 total - the part actually at eye level, and the only part that matters for
    "does this look like it struck the target's feet" - so whether the ground connection
    was even visible varied strike to strike by pure chance. Concentrating most of the
    particle budget into a short, dense lower segment (smite_bolt) makes that connection
    reliable every time, while a separate, sparser segment (smite_bolt_sky) still covers
    the tall upper reach for scale - nobody's looking closely at that part anyway.

    Every distance/radius/length value here is authored in SOURCE UNITS (inches, ÷0.0254
    for real metres - see particle_system.py's own Effect.scale property and
    dissolve_sparks.py's own docstring for the full explanation)."""
    return ParticleSystemDef(
        name,
        {"max_particles": max_particles, "material": _BOLT_MATERIAL, "radius": 1.0, "color": (255, 255, 255, 255)},
        emitters=[_fn("emit_instantaneously", num_to_emit=num_to_emit)],
        initializers=[
            # Horizontal jitter (the first two components) is deliberately tight (real
            # ~8cm) - a wide independent-per-particle scatter reads as several separate
            # strikes landing in a loose cluster instead of one bolt converging exactly on
            # the target's feet (Minecraft's own lightning is a single, essentially
            # straight vertical shaft, not a spray).
            _fn("Position Modify Offset Random", **{
                "offset min": (-3.0, -3.0, z_min), "offset max": (3.0, 3.0, z_max),
                "offset in local space 0/1": True,
            }),
            # A near-vertical plunge fast enough to cross the whole column in a couple of
            # simulated ticks - reads as an instantaneous strike, same as the real thing,
            # rather than something visibly falling. (Already in Source units correctly -
            # real 35-66 m/s.)
            _fn("Velocity Random", speed_in_local_coordinate_system_min=(-3.0, -3.0, -2600.0),
                speed_in_local_coordinate_system_max=(3.0, 3.0, -1400.0)),
            # Narrow so every particle is roughly in sync - appearing and fading together
            # as ONE strike instead of flickering as a staggered handful of segments
            # popping in and out at different times.
            _fn("Lifetime Random", lifetime_min=0.14, lifetime_max=0.18),
            # Real ~0.23-0.46m thick - a tall column reads thin/wispy any thinner; this
            # keeps it reading as one gigantic bolt rather than a string of narrow threads.
            _fn("Radius Random", radius_min=9.0, radius_max=18.0),
            _fn("Sequence Random", sequence_min=0, sequence_max=0),
            # A plain duration (not scaled - see Trail Length Random's own particle_system.py
            # docstring), long enough that the raw speed*trail_time length always exceeds the
            # render min/max length clip below, which is what actually decides the visible
            # streak length.
            _fn("Trail Length Random", length_min=0.15, length_max=0.35),
            _fn("Color Random", color1=(180, 210, 255, 255), color2=(255, 255, 255, 255)),
            _fn("Alpha Random", alpha_min=230, alpha_max=255),
        ],
        operators=[
            # Narrow, same "one synchronized strike" reasoning as the lifetime range above -
            # the whole bolt fades out together, not piecemeal.
            _fn("Alpha Fade Out Random", **{"fade out time min": 0.06, "fade out time max": 0.08,
                                            "proportional 0/1": True}),
        ],
        # min/max length are ALSO in Source units (see this function's own docstring) -
        # real ~1.4-8.1m per streak, scaled to match the column's own thickness above so
        # individual segments don't look stringy relative to it.
        renderers=[_fn("render_sprite_trail", **{"min length": 55.0, "max length": 320.0,
                                                 "length fade in time": 0.0, "animation rate": 0.0})],
    )


def _bolt_trunk():
    """The lower ~18m of the column (real 0.076-17.8m, ÷0.0254) - dense enough (70
    particles in that short a span) to reliably reach the target's actual feet every
    strike - see _bolt_segment's own docstring for why this is split out from the sparser
    upper reach (_bolt_sky) rather than one system spanning the whole height uniformly."""
    return _bolt_segment("smite_bolt", max_particles=90, num_to_emit=70, z_min=3.0, z_max=700.0)


def _bolt_sky():
    """The rest of the same column, ~18-90m up (real, ÷0.0254) - sparser than
    _bolt_trunk, purely for the "reaches toward the sky" scale; see _bolt_segment's own
    docstring. Height capped at a real ~90m, not literally "the sky" - Camera's own far
    clip plane (Modules/Camera/camera.py, far=100.0, never overridden anywhere - grepped)
    is a hard render-distance ceiling nothing in this engine currently exceeds, and
    shadow_module.py's own far distance follows camera.far too, so pushing THAT out
    instead to go taller would soften every shadow in the game just for this one admin
    command's spectacle. 90m leaves a margin under 100 for a smiter standing off to the
    side of the strike (distance to the bolt's top is then slightly more than its height)
    - about as close to "reaches the sky" as the current render distance allows without a
    global, permanent trade-off elsewhere."""
    return _bolt_segment("smite_bolt_sky", max_particles=50, num_to_emit=40, z_min=700.0, z_max=3543.0)


def _crackle():
    return ParticleSystemDef(
        "smite_crackle",
        {"max_particles": 40, "material": _BOLT_MATERIAL, "radius": 1.0, "color": (255, 255, 255, 255)},
        emitters=[_fn("emit_instantaneously", num_to_emit=28)],
        initializers=[
            # A quick burst radiating out from the strike point (mostly upward, Source-space
            # local Z) once the bolt lands - this is the old smite_bolt's own burst, kept as a
            # separate, smaller system for the crackling-electricity impression at impact.
            _fn("Position Within Sphere Random", distance_min=0.0, distance_max=3.0,
                speed_min=400.0, speed_max=900.0, speed_random_exponent=0.6),
            _fn("Velocity Random", speed_in_local_coordinate_system_min=(0.0, 0.0, 250.0),
                speed_in_local_coordinate_system_max=(0.0, 0.0, 650.0)),
            _fn("Lifetime Random", lifetime_min=0.12, lifetime_max=0.28),
            _fn("Radius Random", radius_min=2.0, radius_max=4.0),
            _fn("Sequence Random", sequence_min=0, sequence_max=0),
            _fn("Trail Length Random", length_min=0.03, length_max=0.08),
            _fn("Color Random", color1=(180, 210, 255, 255), color2=(255, 255, 255, 255)),
            _fn("Alpha Random", alpha_min=220, alpha_max=255),
        ],
        operators=[
            # No real gravity - electricity, not debris - just a slight pull so the arcs curl
            # back in rather than flying dead straight forever.
            _fn("Movement Basic", gravity=(0.0, 0.0, -300.0), drag=0.05),
            _fn("Alpha Fade Out Random", **{"fade out time min": 0.4, "fade out time max": 0.7,
                                            "proportional 0/1": True}),
        ],
        # Real ~0.1-1.3m (÷0.0254, see smite_bolt's own docstring on this system-wide
        # Source-units convention) - the unconverted 0.3/4.0 clipped every streak to under
        # 10cm regardless of how far the particle's own speed*trail_time actually reached.
        renderers=[_fn("render_sprite_trail", **{"min length": 4.0, "max length": 52.0,
                                                 "length fade in time": 0.0, "animation rate": 0.0})],
    )


def register_lightning(manager):
    """Adds smite_flash/smite_bolt/smite_bolt_sky/smite_crackle to a ParticleManager - see
    app.py's own spawn_smite for where they're actually spawned."""
    for definition in (_flash(), _bolt_trunk(), _bolt_sky(), _crackle()):
        manager.definitions[definition.name] = definition
