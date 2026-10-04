"""
The Pencil - a second weapon built on the USP rig, same as GoudaGun (see its
own module docstring for the general shape of this). Most of it is inherited
from USP as-is (fire mode, magazine, third-person world model/animations)
since the gun still rides the pistol's own arms - the gun model/animations,
damage, and (see below) accuracy/recoil are a bolt-action sniper's own,
deliberately nothing like the pistol's.

Pencil.glb is a re-export of the scout sniper rig (c_snip_scout.qc_skeleton -
confirmed by inspecting the file directly: its one skin's 43 joints, one mesh
named "Scope" on a node renamed "Pencil", one material "PencilSniper") with
its OWN four named actions baked into that single file - Idle, Shoot, Reload,
DrawAction - rather than this project's usual one-glb-per-pose convention
(USP's own pistol_idle.glb/pistol_shoot.glb/... are each a separate file).
viewmodel_animations below picks each one out by NAME (see weapons_base.py's
_parse_clip_spec/_load_state_clip's own source_clip docstring, added for
exactly this) instead of relying on "the first clip in the file", which would
otherwise just pick the same one action for every state.

Like Gouda_Anims.glb, this file has no arms mesh of its own (only the gun,
skinned to a full rig for its own posing) - the arms still come from
pistol.glb via viewmodel_gun_model overriding just the gun's source file (see
WeaponsBase.viewmodel_gun_model's own docstring).

Bolt-action: SniperRackAnim.glb (same 43-joint skin 0 rig as Pencil.glb,
confirmed by inspecting the file directly) holds one more action, "Rack" -
the bolt-cycle played in between shots (see WeaponsBase.fire()/update()'s own
_rack_pending handling, added for exactly this). It's a SEPARATE file from
Pencil.glb itself (rather than one more action baked into that file) since it
was exported afterward - _parse_clip_spec/_load_state_clip don't care either
way, a (path, source_clip_name) entry works the same regardless of which file
the named action actually lives in.
"""

from .usp import USP

_PENCIL_FILE = "Assets/Models/Arms/New Folder/Pencil/Pencil.glb"
_RACK_FILE = "Assets/Models/Arms/New Folder/Pencil/SniperRackAnim.glb"
_PENCIL_WM_FILE = "Assets/Models/Arms/New Folder/Pencil/WM_Pencil.glb"


class Pencil(USP):
    name = "Pencil"
    weapon_id = "pencil"
    icon = "Assets/Textures/Icons/Pencil/Pencil.png"
    damage = 60
    fire_interval = 1
    draw_speed = 1
    magazine_size = 5
    fire_sound = "Assets/Audio/Guns/Pencil/Pencil.wav"

    # Accuracy: deliberately BAD at rest (hip-fire) - spread_min alone (the cone's
    # size with the trigger rested, before any per-shot penalty) is already most of
    # USP's own spread_max. Scoping in (see has_scope/scope_accuracy_multiplier below)
    # brings it all the way down to PERFECTLY accurate instead - the entire reason to
    # use this gun's scope at all, rather than just a visual flourish. spread_per_shot/
    # recovery barely matter in practice either way: fire_interval plus the mandatory
    # rack (see viewmodel_one_shot_states below) already keep this weapon well under
    # one shot a second, nowhere near fast enough to stack a second shot's penalty
    # before the first's has long since recovered.
    spread_min = 6
    spread_max = 15
    spread_per_shot = 3
    spread_recovery = 5.0
    spread_recovery_delay = 0.15
    # 0.0 - perfectly accurate (spread_degrees() scales straight to 0) once FULLY
    # scoped (eased in alongside everything else about the transition - see
    # WeaponsBase.spread_degrees' own use of this).
    scope_accuracy_multiplier = 0.0

    # Recoil: a huge kick, nothing like USP's light pistol snap - mostly
    # vertical (recoil_pitch) with a noticeable sideways wander
    # (recoil_yaw), arriving fast (recoil_kick_speed) and settling slowly
    # (recoil_recovery, well under USP's own 7.0) so it's still visibly
    # climbing back down through the rack animation's own 0.5s-ish length,
    # not gone before the next shot's even possible.
    recoil_pitch = 14.0
    recoil_yaw = 3.0
    recoil_max = 20.0
    recoil_kick_speed = 220.0
    recoil_recovery = 14

    # Scope: the first (and so far only) weapon with has_scope True - see
    # WeaponsBase.has_scope/scope_point/update_scope for the general
    # mechanism. v_weapon.scout_scope is a real, dedicated scope-eyepiece
    # bone (confirmed present in Pencil.glb's own skin joint list, 44
    # joints total). viewmodel_scope_offset pulls the actual point back a
    # bit FROM that bone (toward the shooter, along its local +Z - confirmed
    # empirically: nudging the offset along each local axis in turn and
    # checking which one brought scope_point() closer to the camera) rather
    # than using the bone's exact position, which sat right at the very
    # front of the eyepiece - a real scope is looked INTO a little, not
    # from its rim. ~3 local units here works out to roughly 6cm in world
    # space at this rig's scale (confirmed by measuring the actual world-
    # space shift, not assumed).
    has_scope = True
    viewmodel_scope_bone = "v_weapon.scout_scope"
    viewmodel_scope_offset = (0.0, 0.0, 1.5)
    scope_time = 0.25
    # A real sniper-grade zoom, well beyond the base class's own generic 20 -
    # once fully scoped (see WeaponsBase.scoped/app.py's own FOV easing),
    # the camera narrows down to this.
    scope_fov = 10.0
    scope_zoom_in_sound = "Assets/Audio/Guns/Pencil/sniper_zoomin.wav"
    scope_zoom_out_sound = "Assets/Audio/Guns/Pencil/sniper_zoomout.wav"

    # v_weapon.scout__Muzzle - a real, dedicated muzzle/barrel-tip bone (confirmed
    # present in Pencil.glb's own skin joint list, 45 joints total - added the same way
    # as the scope bone above). Without this, muzzle_position() was falling back to the
    # world model's own muzzle bone (or, with no gun model posed yet, straight off the
    # camera eye - see that method's own docstring), neither of which is actually where
    # this gun's barrel is.
    viewmodel_muzzle_bone = "v_weapon.scout__Muzzle"

    # WM_Pencil.glb - a dedicated third-person world model (one mesh node
    # "Pencil", one material "PencilSniper", confirmed via a raw glTF dump)
    # instead of inheriting USP's own pistolWM.glb, which would otherwise
    # show every other player a PISTOL in a Pencil-wielder's hand. Rigged to
    # the SAME 4-joint w_pist_usp.qc_skeleton as USP's own pistol world
    # model (confirmed via the same raw glTF dump), so it attaches to
    # worldmodel_bone exactly like USP's does - no bone/skin override
    # needed. worldmodel_position/rotation/scale are left at the base
    # class's own pistol-tuned defaults for now since this is a visibly
    # different (larger, rifle-shaped) model - likely needs its own
    # in-engine tuning to actually sit right in the hand rather than just
    # happening to line up by coincidence.
    worldmodel = _PENCIL_WM_FILE
    # No real animation in the file (its one "animation" entry is just a
    # single-frame bind-pose export artifact, confirmed via the same glTF
    # dump) - same static-pose handling USP's own pistolWM.glb already uses.
    worldmodel_animations = {"idle": None}

    viewmodel_gun_model = _PENCIL_FILE
    # Pencil.glb has only ONE skin (index 0) - unlike pistol.glb's separate
    # arms(0)/gun(1) split, confirmed by inspecting the file directly.
    viewmodel_gun_skin = 0
    viewmodel_animations = dict(
        USP.viewmodel_animations,
        idle=(_PENCIL_FILE, "Idle"),
        shoot=(_PENCIL_FILE, "Shoot"),
        reload=(_PENCIL_FILE, "Reload"),
        draw=(_PENCIL_FILE, "DrawAction"),
        # SniperRackAnim.glb's own skin is ALSO index 0 - still given explicitly here
        # (rather than relying on viewmodel_gun_skin's own default) since that's a
        # property of this ONE file, not a rule to lean on by coincidence.
        rack=(_RACK_FILE, 0, "Rack"),
    )
    # "rack" plays once (from fire(), after "shoot" finishes - see WeaponsBase.update())
    # and drops back to idle, same as every other one-shot state here - not a looping
    # ready/idle-with-bolt-cycle animation.
    viewmodel_one_shot_states = USP.viewmodel_one_shot_states + ("rack",)
