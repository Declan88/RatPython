"""
The Corroder - currently an exact copy of the Pencil (see pencil.py, which has
the long-form notes on how its rig, animations, scope and bolt-action rack
work). Deliberately a standalone copy of the Pencil's settings rather than a
subclass of it, so tuning either weapon never changes the other.

First person now uses its OWN model (an M4-style rig: arms + the Battery/Body
gun parts in one skin, with its own named animations in Corroder.glb). The arms
still come from pistol.glb (so fur tint works) and the gun parts are loaded
from CorroderReload.glb by node, exactly like the Gouda gun does. The third-person
world model is its own too (CorroderWM.glb, on the same w_pist_usp rig as the
Pencil's).
"""

from .usp import USP
from .damage_classes import Zap
from .weapons_base import RIFLE_HOLD_POSE, FireMode

_CORRODER_FILE = "Assets/Models/Arms/New Folder/Corroder/Corroder.glb"
_RELOAD_FILE = "Assets/Models/Arms/New Folder/Corroder/CorroderReload.glb"
_CORRODER_WM_FILE = "Assets/Models/Arms/New Folder/Corroder/CorroderWM.glb"


class Corroder(USP):
    name = "The Corroder"
    weapon_id = "corroder"
    # Its own clip names on the player rig: the USP's "pistol" would make its arm pose the USP's pistol_idle.
    animation_prefix = "corroder"
    icon = "Assets/Textures/Icons/Corroder/Corroder.png"
    # Third person: a rifle, held two-handed in the rifle pose rather than the USP's pistol pose (whose Spine4
    # corrections are tuned for that pose only, so they're cleared here).
    player_animations = {"idle": RIFLE_HOLD_POSE}
    player_upper_rotation_degrees = {}
    player_moving_yaw_degrees = 0.0
    damage = 7.5  # well under the USP's 15: 14 body shots (1.3 s of held fire) to kill (was 12 -> 9 shots, 0.8 s)
    headshot_multiplier = 1.8  # head = 13.5, so 8 head shots (0.7 s) - the base multiplier of 2.0 would make it 7
    fire_mode = FireMode.AUTO  # hold the trigger to keep firing
    fire_interval = 0.1  # 10 rounds/second, same cap as the USP
    draw_speed = 1
    magazine_size = 30
    damage_class = Zap  # kills dissolve the body instead of gibbing it
    tracer_style = "lightning"  # Modules/Graphics/tracer_styles/lightning.py - set here, not on the base
    # Electric-blue muzzle flash (the base's is orange-yellow), to match the lightning tracer.
    muzzle_color = ((60, 140, 255), (200, 230, 255))
    fire_sound = "Assets/Audio/Guns/Corroder/CorroderShot.wav"

    # Accuracy: opens with the USP's own tight first-shot cone (0.4 degrees) with a
    # lower cap (3.0 vs the USP's 3.5), and each shot widens it by about a sixth of what
    # the USP's does (0.12 vs 0.75). Firing faster than spread_recovery_delay never
    # recovers, so sustained fire climbs ~0.12 degrees a shot: about 22 shots to reach
    # the cap, against the USP's 5. Recovery is quicker than the USP's (8 vs 6 degrees a
    # second) so short bursts with a pause between them stay accurate at range.
    spread_min = 0.4
    spread_max = 3.0
    spread_per_shot = 0.12
    spread_recovery = 8.0
    spread_recovery_delay = 0.12
    scope_accuracy_multiplier = 0.0

    # Recoil: a little under the USP's (1.6 / 0.3 / 6.0), same kick and settle speed.
    recoil_pitch = 1.3
    recoil_yaw = 0.25
    recoil_max = 5.0
    recoil_kick_speed = 140.0
    recoil_recovery = 7.0

    # Scope: Corroder.glb has no scope bone, so the sight point is an offset from
    # m4_Parent (bone units: 1 = 2cm) to just above and behind the rear of the
    # body, on the barrel's centre line - a first estimate, tune it in-game.
    has_scope = False
    viewmodel_scope_bone = "v_weapon.m4_Parent"
    viewmodel_scope_offset = (0.91, 10.62, 2.81)
    scope_time = 0.25
    scope_fov = 10.0
    scope_zoom_in_sound = "Assets/Audio/Guns/Pencil/sniper_zoomin.wav"
    scope_zoom_out_sound = "Assets/Audio/Guns/Pencil/sniper_zoomout.wav"

    # No muzzle bone in Corroder.glb either: the offset runs from m4_Silencer out to
    # the barrel tip (found from the mesh's furthest-forward vertices).
    # Pushed 6cm forward (camera space, -Z is forward - see WeaponsBase.viewmodel_offset).
    viewmodel_offset = (0.0, 0.0, -0.06)

    viewmodel_muzzle_bone = "v_weapon.m4_Silencer"
    viewmodel_muzzle_offset = (-0.16, -1.5, -21.71)

    # CorroderWM.glb: one unskinned mesh node ("Battery") under the w_pist_usp rig's root, so it loads as a rigid
    # prop on that rig (muzzle bone ValveBiped.flash included); its two materials share the node, so it's split
    # into one object per material to keep both textures.
    worldmodel = _CORRODER_WM_FILE
    worldmodel_materials = ("Battery", "Body")
    worldmodel_animations = {"idle": None}

    # Gun parts come from CorroderReload.glb's single skin, one object per mesh node
    # (a skinned object keeps only one material). Same rig, parts and textures as
    # Corroder.glb, but with the Body's corrected normals. Its c_arms node is skipped
    # on purpose - the arms come from pistol.glb (WeaponsBase.viewmodel_model).
    viewmodel_gun_model = _RELOAD_FILE
    viewmodel_gun_skin = 0
    viewmodel_gun_nodes = ("Battery", "Body")
    # The file bakes 17 named actions; the plain ones are used, not the *_unsil
    # (silencer-off) variants, and shoot1 of the three shoot variants.
    viewmodel_animations = dict(
        USP.viewmodel_animations,
        idle=(_CORRODER_FILE, "idleAction"),
        shoot=(_CORRODER_FILE, "shoot1Action"),
        reload=(_RELOAD_FILE, "reloadAction"),  # its own file, same rig as Corroder.glb
        draw=(_CORRODER_FILE, "drawAction"),
    )
    # No "rack" clip exists for this model, so no bolt-cycle between shots (the
    # Pencil's own SniperRackAnim doesn't fit this rig).
