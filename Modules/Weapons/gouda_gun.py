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

from .usp import USP

_GOUDA_DIR = "Assets/Models/Arms/New Folder/Gouda"


class GoudaGun(USP):
    name = "Gouda Gun"
    icon = "Assets/Textures/Icons/Gouda/Gouda.png"

    viewmodel_gun_model = f"{_GOUDA_DIR}/Gouda_Anims.glb"
    viewmodel_gun_skin = 0

    viewmodel_animations = dict(
        USP.viewmodel_animations,
        idle=(f"{_GOUDA_DIR}/Gouda_Idle.glb", 0),
        shoot=(f"{_GOUDA_DIR}/Gouda_fire.glb", 0),
    )
