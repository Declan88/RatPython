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

    # Accuracy: a semi-auto pistol. First shot from rest is tight; each shot
    # opens the cone by 0.75 degrees, and it closes again at 6 degrees a second
    # after a short pause - so a steady ~4 shots a second holds its accuracy
    # (0.75 added per shot vs about 0.8 recovered between them), and only
    # clicking much faster than that opens it up, slowly, toward the cap.
    spread_min = 0.4
    spread_max = 3.5
    spread_per_shot = 0.75
    spread_recovery = 6.0
    spread_recovery_delay = 0.12

    # Recoil: a light kick with a little sideways wander, settling in about a
    # quarter of a second - a pistol snaps up and comes right back.
    recoil_pitch = 1.6
    recoil_yaw = 0.3
    recoil_max = 6.0
    recoil_kick_speed = 140.0
    recoil_recovery = 7.0

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
