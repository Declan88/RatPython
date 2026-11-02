"""
The Farm Sweeper - a two-shot, semi-automatic double barrel: 100 damage a shot, so one body shot kills.

First person uses FarmSweeperArms.glb (the scout rig - arms plus the gun's parts in one skin, with its own named
actions Equip/Idle/Reload/Shoot), loaded like the Corroder: the arms still come from pistol.glb (so fur tint works)
and only the gun's own nodes are taken from the file - it also carries leftover Pencil/scout scope meshes, which
are skipped. Its sounds are in Assets/Audio/Guns/FarmSweeper, its icon is Assets/Textures/Icons/Farm/Farm.png. No
world model of its own yet: third person shows the USP's pistol model until one is made.
"""

from .usp import USP
from .weapons_base import RIFLE_HOLD_POSE, AltFire, FireMode

_FARM_SWEEPER_FILE = "Assets/Models/Arms/New Folder/Farm Sweeper/FarmSweeperArms.glb"
_SOUND_DIR = "Assets/Audio/Guns/FarmSweeper"


class FarmSweeper(USP):
    name = "Farm Sweeper"
    weapon_id = "farm_sweeper"
    # Its own clip names on the player rig: the USP's "pistol" would make its arm pose the USP's pistol_idle.
    animation_prefix = "farm_sweeper"
    icon = "Assets/Textures/Icons/Farm/Farm.png"
    damage = 100                     # the whole load: about 11.1 a pellet
    # A 12 gauge 00 buckshot shell holds 9 pellets, landing within 4 degrees of the aim (about 1.4 m across at
    # 10 m) - tighter than HL2's shotgun cone (5), still wider than real buckshot's, which only spreads about an
    # inch per yard.
    pellets = 9
    pellet_spread = 4.0
    impact_size = 0.5                # small pellet holes, not 9 full-size bullet impacts
    fire_mode = FireMode.SEMI        # one shot per trigger pull
    fire_interval = 0.3
    magazine_size = 2                # both barrels, then reload
    draw_speed = 1                   # the Equip animation at its own speed (the USP plays its draw 3x)

    fire_sound = f"{_SOUND_DIR}/Shot.wav"
    # Reload sounds at Reload's own frames (30 fps): break it open, load each barrel, snap it shut. Replaces the
    # USP's clip-out/clip-in/slide sounds, which this would otherwise inherit.
    handling_sounds = {
        "reload": {
            12: f"{_SOUND_DIR}/Open.wav",
            30: f"{_SOUND_DIR}/Load.wav",
            45: f"{_SOUND_DIR}/Load.wav",
            62: f"{_SOUND_DIR}/Close.wav",
        },
    }

    # Recoil: a big kick, about the Pencil's (the USP's is 1.6 / 0.3 / 6.0, the Pencil's 14 / 3 / 20), that snaps up
    # and comes straight back down in time with the Shoot animation (0.4 s): the view peaks about 25 ms after the
    # shot (kick speed) and is level again at 14 / 35 = 0.4 s (recovery). Retune the recovery with the kick
    # (pitch / animation length) to keep it in step. The alt fire's is the stronger version.
    recoil_pitch = 14.0
    recoil_yaw = 4.0
    recoil_max = 28.0
    recoil_kick_speed = 700.0
    recoil_recovery = 35.0
    fire_rumble = (0.6, 0.5, 0.15)   # a heavy thump, well over a pistol's (bindings.RUMBLE_FIRE)

    # Alt fire: both barrels at once - two full loads (18 pellets, 200 damage) along the same aim, using both rounds,
    # with a stronger kick, and it throws you 8 m/s back the way you aimed: at the floor that's a launch about
    # 3 m up (a jump is 6 m/s - fired mid-jump the two add up). Both barrels going off are much louder (the shot
    # tripled, soft-clipped so it doesn't crackle: about +4.5 dB, as the file already peaks at full scale) and carry
    # further. A kill with it blows the victim's gibs away from you at 14 m/s (a Gouda Gun blast's are 10 at most).
    # With one barrel left it fires that one as an ordinary shot, without the launch or the gib throw.
    alt_fire = AltFire(
        projectiles=[{}, {}],
        ammo_per_shot=2,
        fallback_to_primary=True,
        # Recoil in step with the same 0.4 s Shoot animation: back level at 18 / 45 = 0.4 s.
        stats={"recoil_pitch": 18.0, "recoil_yaw": 5.0, "recoil_max": 36.0,
               "recoil_kick_speed": 700.0, "recoil_recovery": 45.0,
               "fire_rumble": (1.0, 0.8, 0.22),
               "self_knockback": 8.0, "fire_gain": 3.0, "fire_max_distance": 220.0, "gib_push": 14.0},
    )

    # A long gun: the two-handed rifle pose in third person, without the USP's pistol-pose corrections.
    player_animations = {"idle": RIFLE_HOLD_POSE}
    player_upper_rotation_degrees = {}

    # Seen as through the Blender camera it was animated with (FarmSweeperArms.blend): 50 mm lens, 36 mm sensor,
    # 1920x1080, fit Auto - 39.5978 deg horizontal - sitting 0.3986 m behind the rig's origin, so the model is drawn
    # with that FOV and pushed that far out in front of the game camera (camera space, -z is forward).
    viewmodel_fov = 39.5978
    # ...then lowered 3 cm so it sits a little lower on screen than the Blender framing.
    viewmodel_offset = (0.0, -0.03, -0.3986)
    # Sway around the grip (the v_weapon.scout_Parent bone: about 0.30 m ahead of the rig origin and 5 cm down, so
    # about 0.70 m from the eye and 8 cm down once shifted above), not the default 0.35 m, which is behind it.
    viewmodel_sway_pivot = (0.0, -0.08, -0.70)
    viewmodel_gun_model = _FARM_SWEEPER_FILE
    viewmodel_gun_skin = 0
    viewmodel_gun_nodes = ("Barrel", "GunBody", "Shell1", "Shell2")
    viewmodel_animations = dict(
        USP.viewmodel_animations,
        idle=(_FARM_SWEEPER_FILE, "Idle"),
        shoot=(_FARM_SWEEPER_FILE, "Shoot"),
        reload=(_FARM_SWEEPER_FILE, "Reload"),
        draw=(_FARM_SWEEPER_FILE, "Equip"),
    )
    # The flash and tracer start at the barrels' mouth: from the barrel bone along the barrel to the middle of the
    # Barrel mesh's front-most vertices (bone units, 1 = 2 cm - so about 16 cm out). Estimated from the model, not
    # tuned in game.
    viewmodel_muzzle_bone = "v_weapon_barrel"
    viewmodel_muzzle_offset = (0.0, 0.70, -8.22)
