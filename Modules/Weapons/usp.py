"""The USP pistol - the first weapon. Everything it needs is inherited from
WeaponsBase's pistol defaults; the class attributes are spelled out anyway so
this file is the one place that says exactly which assets make up the USP."""

from .weapons_base import WeaponsBase, _POSE_DIR


class USP(WeaponsBase):
    name = "USP"
    animation_prefix = "pistol"

    # Gunfire
    fire_sound = "Assets/Audio/Guns/USP/usp_unsil-1.wav"
    damage = 15.0

    # First person: pistol.glb holds the arms (skin 0) and the gun (skin 1). The
    # gun rig is the arms rig plus the weapon bones, so its clips play on both.
    viewmodel_model = f"{_POSE_DIR}/pistol.glb"
    viewmodel_arms_skin = 0
    viewmodel_gun_skin = 1
    viewmodel_animations = {
        "idle": f"{_POSE_DIR}/pistol_idle.glb",
        "shoot": f"{_POSE_DIR}/pistol_shoot.glb",
    }
    viewmodel_one_shot_states = ("shoot",)

    # Third-person: the model held in the character's hand (its own baked clip
    # is the idle), and the pose of the character's arms holding it.
    worldmodel = f"{_POSE_DIR}/pistolWM.glb"
    worldmodel_animations = {"idle": None}
    player_animations = {"idle": "Assets/Animations/Poses/Pistol/pistolidle.glb"}
    # Tuned in-game against this pose.
    player_upper_rotation_degrees = {"ValveBiped.Bip01_Spine4": (-10, 60, 10)}
    player_moving_yaw_degrees = -20.0
