"""
A Quake-railgun-style spiral of star sprites winding around a weapon's own tracer beam. Any
weapon can have one:

    tracer_spiral = TracerSpiral(color=(255, 210, 30))

When it fires, WeaponsBase.spawn_tracer_spiral (called from app.py right after the tracer
itself, see maybe_spawn_tracer_spiral) hands this straight to ParticleManager.spawn_beam_
spiral, which does the actual particle maths against whatever values THIS instance carries -
so a different weapon wanting a tighter, denser, bigger or differently-coloured spiral just
makes its own TracerSpiral with different numbers; nothing outside this file and the one
spawn_beam_spiral call needs to change.

This file is only the parameters - the maths (the helix shape, the camera-distance coverage
falloff for a beam too long to spiral at full density) lives in Modules/Particles/
particle_system.py's own spawn_beam_spiral, same split Explosion.py's own docstring
describes for explosions ("this file is only the maths... stays in app.py" - the same idea,
mirrored: this file is only the NUMBERS, the maths lives with the particle system).
"""


class TracerSpiral:
    def __init__(self, color=None, material="particle/particle_glow_04", turns_per_metre=0.25,
                 particles_per_metre=2.0, max_particles=450, radius=0.05, star_size=0.05,
                 lifetime=0.4):
        """color: (r, g, b) 0-255, or None for the star sprite's own natural (white/additive)
        colour. material: the sprite each star renders as - the small round glow used
        elsewhere as "the star" by default (particle/star's own alias - see impacts.py's
        MATERIAL_ALIASES docstring: no real star image was in the exported set).

        turns_per_metre: the helix's own tightness - CONSTANT regardless of beam length, so a
        long shot doesn't read as a loosely-stretched spiral (or a short one as an over-tight
        coil): total turns scale with the beam's actual length instead of being fixed.

        particles_per_metre/max_particles: the spiral's density and the total particle budget
        it's capped at - together they set how much of a very long beam gets covered at full
        density (max_particles / particles_per_metre = the reach, in metres) before the
        spiral simply STOPS partway along the beam (plain tracer, no spiral, beyond that)
        rather than silently thinning out over the whole length - see spawn_beam_spiral's own
        docstring for why that matters.

        radius: metres out from the beam's own centreline. star_size: each sprite's own
        radius, metres. lifetime: seconds each star takes to fade out."""
        self.color = color
        self.material = material
        self.turns_per_metre = turns_per_metre
        self.particles_per_metre = particles_per_metre
        self.max_particles = max_particles
        self.radius = radius
        self.star_size = star_size
        self.lifetime = lifetime
