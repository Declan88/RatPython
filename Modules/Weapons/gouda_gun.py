"""
The Gouda Gun - a second weapon SLOT, replacing the USPAuto placeholder now
that there's a real second model and icon for it. Everything about it is
inherited from USP as-is (fire rate, damage, magazine, draw speed, muzzle
bone, third-person world model/animations) since it's built on the exact
same rig (c_pist_usp.qc_skeleton) and reuses USP's own pistol_shoot/pistol_
reload/pistol_draw clips - only the gun model, icon, and idle clip differ.

Gouda_Anims.glb holds ONLY the gun (its own Cheese/glass materials, no arms
of its own) - the arms still come from pistol.glb, same as USP, via
viewmodel_gun_model overriding just the gun's source file (see
WeaponsBase.viewmodel_gun_model's own docstring). An earlier export
(Gouda_Gun.glb) baked the arms and gun into one combined mesh/skin, which
made the whole model pick up the arms' rat-fur material instead of the
gun's own (skeletal_loader.load_skinned_glb only keeps one material per
skinned object - see its own multi-material warning) - this fixes that by
keeping the gun as its own separately-materialed object again.

Idle and shoot each use their own clip (Gouda_Idle.glb, Gouda_fire.glb)
instead of USP's pistol_idle.glb/pistol_shoot.glb: the Gouda gun mesh is
rigidly weighted 100% to one bone (v_weapon.USP_Parent), unlike the
pistol's own gun mesh (split across several bones, most of which don't
move during idle/shoot), so USP's clips' motion on that one bone - barely
visible on the pistol - swept the whole, bulkier cheese-wedge mesh
noticeably sideways. Both Gouda_*.glb files are re-authored clips baked
for this mesh/weighting instead (reload/draw haven't needed the same
treatment yet).
"""

from .explosion import Explosion
from .tracer_spiral import TracerSpiral
from .usp import USP

_GOUDA_DIR = "Assets/Models/Arms/New Folder/Gouda"


class GoudaGun(USP):
    name = "Gouda Gun"
    weapon_id = "gouda_gun"
    icon = "Assets/Textures/Icons/Gouda/Gouda.png"
    tracer_style = "laser"
    damage = 5.0                     # direct hit
    fire_interval = 0.23             # a bit slower than the USP's 0.1 (reverted an earlier 0.17
                                      # tweak, then nudged up again from 0.2 for a bit more rest
                                      # between shots - shots-to-kill is unaffected, since that's
                                      # set by damage, not interval; body TTK moves to (5-1)*0.23
                                      # = 0.92s)
    # radius unchanged at 5.0 (was 3.0 - still a real reach increase from the original "too
    # weak" report). damage brought back down from 25.0 to 19.0: every landed shot ALSO deals
    # its own splash to the same target it hit (near point-blank range, strength ~0.85-0.95 -
    # see Explosion.strength and app.py's own explode()), so total effective damage per body
    # shot is direct(5) + splash(19 * ~0.85-0.95 ~= 16-18) = ~21-23 - comfortably in the
    # ceil(100/x)=5 bracket (any effective damage in [20, 25) kills in exactly 5 body shots),
    # not 25.0's own ~27-28 effective (4 shots, tied with the USP).
    explosion = Explosion(radius=5.0, damage=19.0, self_damage_fraction=0.5, push_speed=10.0)
    fire_sound = "Assets/Audio/Guns/Gouda/GoudaBlast.wav"
    # The raygun's effects, recoloured yellow (only these two systems are loaded from the file).
    particle_files = {"Assets/Particles/blast/bo3_raygun.pcf": ["bo3_raygun_impact", "bo3_raygun_muzzleflash"]}
    impact_particle = "bo3_raygun_impact"
    impact_color = ((255, 210, 30), (255, 245, 120))
    # A Quake-railgun-style spiral of star sprites winding around the laser beam - see
    # tracer_spiral.py's own docstring - in the same yellow as everything else this gun does.
    # Every other tunable (tightness, density, size, lifetime) is left at TracerSpiral's own
    # default, which is exactly this spiral's already-tuned look - only the colour is Gouda's
    # own business here.
    tracer_spiral = TracerSpiral(color=impact_color[0])
    # The raygun's own muzzle flash (a separate system in the same .pcf), in the same yellow.
    muzzle_particle = "bo3_raygun_muzzleflash"
    muzzle_color = impact_color
    muzzle_size = 0.35     # the USP's flash is 0.7
    # The Gouda barrel tip is ~16 cm ahead of the USP silencer bone the flash is attached to.
    viewmodel_muzzle_offset = (-0.03, -1.02, -7.84)

    viewmodel_gun_model = f"{_GOUDA_DIR}/Gouda_Anims.glb"
    viewmodel_gun_skin = 0
    # Body, cheese and glass each keep their own material.
    viewmodel_gun_nodes = ("GoudaGun_Body", "GoudaGun_Cheese", "GoudaGun_Glass", "GoudaGun_Text")
    # Body is hard-cutout (masked); glass is alpha-blended.
    viewmodel_alpha_mode_overrides = {"RefBake": "MASK", "Glass": "BLEND"}
    # The text shares RefBake with the body but ignores its mask override (keeps the authored mode).
    viewmodel_node_alpha_mode_overrides = {"GoudaGun_Text": None}

    viewmodel_animations = dict(
        USP.viewmodel_animations,
        idle=(f"{_GOUDA_DIR}/Gouda_Idle.glb", 0),
        shoot=(f"{_GOUDA_DIR}/Gouda_fire.glb", 0),
    )

    # Third-person model (what other players/a mirror would see) - same per-material node
    # split as the viewmodel gun, in this file's own node names.
    worldmodel = f"{_GOUDA_DIR}/WM_Gouda.glb"
    worldmodel_nodes = ("Gouda_Body", "Gouda_Cheese", "Gouda_Dome", "Gouda_Text")
    worldmodel_alpha_mode_overrides = {"RefBake": "MASK", "glass/glasswindow007a": "BLEND"}
    # Text shares RefBake with the body but ignores its mask override (keeps the authored mode) -
    # same reasoning as viewmodel_node_alpha_mode_overrides.
    worldmodel_node_alpha_mode_overrides = {"Gouda_Text": None}
