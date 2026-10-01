"""
Bullet impact effects: which particle system (impact_fx.pcf) a shot that hits
a surface plays, chosen by the surface's physical material name (the same names
footsteps use - see Assets/Audio/footsteps and Scene.add_static's
physical_material).
"""

from Modules.Audio.bullet_impact_sounds import get_bullet_impact_sound

IMPACT_FILE = "Assets/Particles/Impact/impact_fx.pcf"
# A bullet hitting a surface is a short, punchy one-shot, not a weapon's own
# report (see WeaponsBase.fire_min_distance/fire_max_distance for that, much
# louder/further-carrying) - audible up close, inaudible from across the map.
# Same shape of pair gibs.py's own GORE_SOUND uses for its impact, close
# enough in kind (a sudden, similarly-loud one-shot) to reuse that distance.
IMPACT_SOUND_MIN_DISTANCE = 3.0
IMPACT_SOUND_MAX_DISTANCE = 60.0
# Impact effects (bullet hits, a weapon's impact_particle) count against one cap so a
# burst of shots can't pile up unlimited particle systems: see ParticleManager.group_limits.
IMPACT_GROUP = "impact"
MAX_IMPACT_EFFECTS = 12

DEFAULT_IMPACT = "impact_concrete"

# Which texture (under Assets/Particles/ParticleIMGs) each material in the impact
# file uses when its .vmt isn't around to say - guesses from the names, meant to be
# replaced by the real $basetexture if a .vmt is added beside the images.
MATERIAL_ALIASES = {
    "particle/smoke1/smoke1_fade": "particle/smoke1/smoke1",
    "particle/particle_glow_04_additive": "particle/particle_glow_04",
    "particle/particle_debris_burst/particle_debris_burst_001": "particle/particle_debris_01",
    "particle/particle_debris_burst/particle_debris_burst_002": "particle/particle_debris_02",
    "particle/vistasmokev1_min_depth_nearcull": "particle/vistasmokev1/vistasmokev1",
    "particle/water_splash/water_splash": "particle/splash01/splash01",
    # Not in the exported images: stand-ins (a single wood chip for the fleck sheet, a
    # round glow for the star). Replace with the real textures if found.
    "particle/impact/fleks": "fleck_wood1",
    "particle/star": "particle/particle_glow_04",
}

SURFACE_IMPACTS = {
    "concrete": "impact_concrete",
    "tile": "impact_concrete",
    "wood": "impact_wood",
    "woodpanel": "impact_wood",
    "ladder": "impact_wood",
    "metal": "impact_metal",
    "metalgrate": "impact_metal",
    "chainlink": "impact_metal",
    "duct": "impact_metal",
    "dirt": "impact_dirt",
    "grass": "impact_dirt",
    "gravel": "impact_dirt",
    "mud": "impact_dirt",
    "sand": "impact_dirt",
    "snow": "impact_dirt",
    # Water: no puff of debris.
    "wade": None,
    "slosh": None,
}


# The impact file is authored in Source's inches for a game where a player is ~1.8 m;
# these bring it down to size here. IMPACT_SIZE scales everything about an impact
# (particle radii, spread, speeds); SMOKE_RADIUS_SCALE shrinks just the smoke and dust
# sprites (whose Source radii are up to ~1.3 m) further.
IMPACT_SIZE = 0.6
SMOKE_RADIUS_SCALE = 0.5
RADIUS_SCALE = {
    "particle/smoke1/smoke1": SMOKE_RADIUS_SCALE,
    "particle/smoke1/smoke1_fade": SMOKE_RADIUS_SCALE,
    "particle/particle_smokegrenade": SMOKE_RADIUS_SCALE,
    "particle/vistasmokev1/vistasmokev1": SMOKE_RADIUS_SCALE,
    "particle/vistasmokev1_min_depth_nearcull": SMOKE_RADIUS_SCALE,
    "particle/water_splash/water_splash": SMOKE_RADIUS_SCALE,
}


# How long smoke lingers: the file has it hang for 2-4 s (3 s for wood's). Fades and
# growth are relative to the lifetime, so this just plays them faster.
SMOKE_LIFETIME_SCALE = 0.35
LIFETIME_SCALE = {
    "particle/smoke1/smoke1": SMOKE_LIFETIME_SCALE,
    "particle/smoke1/smoke1_fade": SMOKE_LIFETIME_SCALE,
    "particle/vistasmokev1/vistasmokev1": SMOKE_LIFETIME_SCALE,
    "particle/vistasmokev1_min_depth_nearcull": SMOKE_LIFETIME_SCALE,
}


def keep_frame(material_key, rgb):
    """ParticleManager frame_filter: vistasmokev1's sprite sheet holds fire
    frames after its smoke ones, and its animated sequences run through them (smoke
    that turns orange). Impacts only want the smoke, so warm frames are dropped."""
    if "vistasmoke" in material_key:
        r, g, b = rgb
        return (r - b) < 30
    return True


def impact_effect_name(material):
    """The particle system for a surface material name (None: no effect for that
    surface; an unknown or unset material gets the concrete one)."""
    if material is None:
        return DEFAULT_IMPACT
    return SURFACE_IMPACTS.get(str(material).lower(), DEFAULT_IMPACT)


def play_impact_sound(sound_manager, hit):
    """Plays a random bullet-impact sound (see bullet_impact_sounds.py)
    for hit.material, positioned at the hit with reasonable attenuation
    (IMPACT_SOUND_MIN_DISTANCE/MAX_DISTANCE above). Separate from
    spawn_impact below so a weapon with its own custom impact_particle
    (e.g. the Gouda Gun's raygun burst - see app.py's own shooter_
    weapon.impact_particle branch) still gets a material-appropriate
    sound without ALSO getting spawn_impact's generic surface debris
    particle on top of its own.

    Called from both of app.py's own shot-handling sites (the shooter's
    immediate local feedback, and remote_shot, which runs identically
    on every OTHER client off the networked tracer event) - the sound
    is "replicated" the same way the particle effects already are: no
    new networking of its own, just reusing the two call sites every
    client already goes through for the particles. Hits on players
    (hit.owner set) play nothing, same as spawn_impact."""
    if sound_manager is None or hit is None or hit.owner is not None:
        return
    sound_path = get_bullet_impact_sound(hit.material)
    if sound_path is None:
        return
    sound_manager.add_sound(
        sound_path, hit.position, loop=False,
        min_distance=IMPACT_SOUND_MIN_DISTANCE, max_distance=IMPACT_SOUND_MAX_DISTANCE,
        falloff="inverse",
    )


def spawn_impact(particles, hit, size=IMPACT_SIZE, sound_manager=None):
    """Plays the impact effect for a RayHit (Modules/Physics/physics_world.py)
    where it struck. Hits on players (hit.owner set) play nothing here.
    sound_manager: see play_impact_sound above - passed through and
    called here too so a caller only needs this one function for the
    common case (no custom weapon impact_particle)."""
    if hit is None or hit.owner is not None:
        return None
    play_impact_sound(sound_manager, hit)
    name = impact_effect_name(hit.material)
    if name is None:
        return None
    return particles.spawn_surface(name, hit.position, hit.normal, size=size, group=IMPACT_GROUP)
