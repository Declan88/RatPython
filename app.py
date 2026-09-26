import sys
import pygame
import os
import ctypes
import glm


def _is_frozen_build():
    """True for a Nuitka-compiled run ("__compiled__" injected into this
    entry module's globals - Nuitka's own documented detection method),
    False for an ordinary `python app.py` dev run. Shared by every
    frozen-only check in this file (asset-path chdir, GPU-preference
    registry key) so they can't silently drift out of sync with each
    other."""
    return "__compiled__" in globals()


def _chdir_for_frozen_build():
    """Nuitka's --onefile mode re-executes itself from its own unpacked
    temp directory before any of this code runs, so sys.executable
    already points at that unpacked location by the time this runs (for
    --standalone too - there's just no separate temp dir to unpack to,
    sys.executable is simply where the built exe already sits). Every
    asset path in this codebase (torus_scene.py's model/texture/audio
    paths, etc.) is a plain relative string like "Assets/Models/...",
    which only resolves correctly if the process's current working
    directory happens to BE that location. Since none of those call
    sites can be reached before this runs (they're all behind imports/
    calls below), chdir'ing here once, before anything else executes,
    makes every existing relative path keep working unmodified instead
    of touching dozens of individual asset-loading call sites."""
    if _is_frozen_build():
        os.chdir(os.path.dirname(os.path.abspath(sys.executable)))


_chdir_for_frozen_build()


def _request_high_performance_gpu():
    """Nvidia Optimus / AMD dynamic-switchable-graphics laptops default
    an arbitrary .exe to the low-power integrated GPU. This registry
    key is a real, documented, vendor-neutral mechanism (Windows 10
    Creators Update+) - the exact one Settings > System > Display >
    Graphics writes when a user manually sets an app to "High
    performance" - but confirmed in practice that it's specifically
    designed for/reliably honored by DXGI (Direct3D) applications, NOT
    a raw OpenGL context created via WGL (which is what this project's
    moderngl/pygame-ce rendering does) - see the NOTE below this
    function for the mechanism that actually works for OpenGL. Kept
    here anyway as a harmless belt-and-suspenders extra: costs nothing,
    and covers any future D3D-based rendering path or driver version
    where it does get honored.

    Keyed by the exact exe PATH (sys.executable), so this only ever
    affects this one built exe, never other unrelated programs. Only
    runs for a frozen/built exe, not a dev `python app.py` run - a
    dev run's sys.executable is
    python.exe itself, and forcing a GPU preference there would apply
    to every OTHER Python script run through that same interpreter
    too, not just this project."""
    if sys.platform != "win32" or not _is_frozen_build():
        return
    try:
        import winreg

        exe_path = os.path.abspath(sys.executable)
        key = winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\DirectX\UserGpuPreference",
            0,
            winreg.KEY_SET_VALUE,
        )
        with key:
            # "GpuPreference=2;" is Windows' own literal value format
            # for this key (2 = high performance, 1 = power saving, 0 =
            # let the system/driver decide) - exactly what the
            # Settings UI itself writes when a user picks "High
            # performance" by hand.
            winreg.SetValueEx(key, exe_path, 0, winreg.REG_SZ, "GpuPreference=2;")
    except OSError as e:
        print(f"Warning: couldn't set GPU preference in registry: {e}")


_request_high_performance_gpu()

# NOTE on Optimus/switchable-graphics GPU selection for OpenGL: the
# registry hint above isn't reliably honored for a raw OpenGL context
# (moderngl/pygame-ce, via WGL) the way it is for Direct3D/DXGI apps.
# The mechanism that DOES work is exporting two symbols -
# NvOptimusEnablement and AmdPowerXpressRequestHighPerformance - but
# these MUST be in the actual .exe's own PE export table specifically;
# a DLL loaded at runtime does nothing, regardless of how early it's
# loaded here - an earlier version of this file tried exactly that (a
# companion gpu_hint.dll) and it had no effect, and the old PyInstaller
# build needed a patched bootloader to get this right for the same
# underlying reason (that workaround was PyInstaller-specific and
# doesn't carry over to Nuitka).
#
# Nothing in THIS Python file can express that fix - there is no
# per-build-invocation flag or Python-level hook for it - but
# nuitka_gpu_preference_plugin.py (project root, loaded via
# BuildCMD_Nuitka's own --user-plugin= flag) now does, using Nuitka's
# getExtraCodeFiles() plugin hook. Confirmed working (see that file's
# own docstring for the pefile-verified proof and the two non-obvious
# details that made it work: which compile stage a --onefile build's
# "onefile_"-prefixed extra-code-file key actually reaches, and why the
# onefile bootstrap .exe - not some separate extracted exe - is the
# right target in the first place).

from Modules.Window.window import WindowManager
from Modules.Window.loading_screen import show_loading_screen
from Modules.UI import UIManager
from Modules.UI.demo import build_demo
from Modules.UI import Anchor, Crosshair, Hitmarker, ProgressBar
from Modules.UI.death_screen import DeathScreen
from Modules.Graphics.tracers import Tracers
from Modules.Particles import ParticleManager
from Modules.Particles.blood import register_blood
from Modules.Gore import GibManager
from Modules.Particles.impacts import IMPACT_FILE, MATERIAL_ALIASES, LIFETIME_SCALE, RADIUS_SCALE, keep_frame, spawn_impact
from Modules.Physics.physics_world import CollisionGroup
from Modules.UI.nametags import NameTags
from Modules.UI.scoreboard import Scoreboard

SPAWN_POSITION = (0.0, 2.0, 3.0)   # where the player (re)spawns: hull centre, metres
RESPAWN_SECONDS = 3.0
HITMARKER_SOUND = "Assets/Audio/Player/hitmarker.wav"
HITMARKER_VOLUME = 1.0   # the sound manager applies a volume twice (on the sound and on its channel), so 1.0 is what plays at full
HITMARKER_HEADSHOT_PITCH = 1.45   # a headshot's hitmarker sound is this much higher (playback speed)
HITMARKER_GAIN = 10.0     # ...so anything louder has to amplify the samples themselves (soft-clipped)
from Modules.Graphics.paper_doll import PaperDoll
from Modules.Camera.camera import Camera
from Modules.Camera.camera_boom_arm import CameraBoomArm
from Modules.Scenes.torus_scene import TorusScene
from Modules.Scenes.mainmap_scene import MainMapScene
from Modules.Scenes.main_menu_scene import MainMenuScene
from Modules.Physics.character_controller import CharacterController
from Modules.Player.player_model import PlayerModel
from Modules.Player.rat_colors import RAT_TINT_MASK_PATH
from Modules.Player.viewmodel import ViewModel
from Modules.Weapons import USP
from Modules.Weapons.weapons_base import WeaponsBase


def load_steam_api_dll():
    # One shared library name per platform - Valve's Steamworks SDK ships
    # a differently-named/formatted binary for each (steam_api64.dll on
    # Windows, libsteam_api.dylib on macOS - a universal x86_64+arm64
    # binary here, so one file covers both Intel and Apple Silicon -
    # libsteam_api.so on Linux). ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)
    # and RTLD_GLOBAL itself are both real cross-platform POSIX/ctypes
    # concepts, not Windows-specific, so the loading logic below is
    # identical for all three - only the filename differs.
    lib_name = {
        "win32": "steam_api64.dll",
        "darwin": "libsteam_api.dylib",
    }.get(
        sys.platform, "libsteam_api.so"
    )  # covers "linux" and any other POSIX platform

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # _chdir_for_frozen_build() (called at module import time, above this
    # function's own call further down) already puts the process's cwd
    # at the right unpacked location for a Nuitka build on every
    # platform, so the os.getcwd() candidate below covers a frozen build
    # on its own - no separate extraction-folder special-case needed.
    candidates = [
        os.path.join(script_dir, lib_name),
        os.path.join(os.getcwd(), lib_name),
    ]

    env_override = os.environ.get("STEAM_API_DLL_PATH")
    if env_override:
        candidates.insert(0, env_override)

    for path in candidates:
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(os.path.dirname(path))
                print(f"Pre-loaded {lib_name} from: {path}")
                return
            except Exception as e:
                print(f"Found {lib_name} at {path} but failed to load it: {e}")

    print(f"Warning: {lib_name} not found.")


# Pre-load the platform's Steamworks shared library globally before any
# modules import py_steam_net.
load_steam_api_dll()

from Modules.Networking.network_manager import NetworkManager


def main():
    print("Yo wsg")
    window = WindowManager(800, 600, "RatWar")
    camera = Camera(position=(0.0, 0.0, 3.0), aspect=window.width / window.height)

    # Scene classes, not instances - see get_or_load_scene below. Only
    # ever constructed the first time something actually switches to
    # that key, not all up front: TorusScene is kept around purely as a
    # K_1-selectable test/comparison scene (see the switch handler
    # below), and building it (parsing its glb, uploading GPU
    # resources, baking lightmaps) is real, non-trivial work worth
    # skipping entirely for a run that never visits it.
    SCENE_CLASSES = {"torus": TorusScene, "mainmap": MainMapScene, "mainmenu": MainMenuScene}
    scenes = {}

    def get_or_load_scene(key):
        """Returns the already-built scenes[key] if it exists, else
        constructs it first - showing one loading-screen frame right
        before that (still fully synchronous/blocking - see loading_
        screen.py's own docstring for why) construction runs, so
        switching to a not-yet-visited scene reads as "loading"
        instead of as a frozen window."""
        if key not in scenes:
            show_loading_screen(window, f"Loading {key}...")
            scenes[key] = SCENE_CLASSES[key](window.ctx)
            scenes[key].gibs = GibManager(scenes[key], particles)
            # See WindowManager.reset_frame_timer's own docstring - a
            # multi-second construction just ran on this same thread,
            # and the NEXT dt computed would otherwise include all of it.
            window.reset_frame_timer()
        return scenes[key]

    # Gameplay state. Nothing below exists until setup_game() runs (the
    # main menu's Play button) - it is what loads mainmap, so the menu
    # doesn't pay for that load up front. The main loop only reaches any of
    # this after in_menu goes False.
    current_scene_key = None
    current_scene = None
    player = None
    player_height = None
    local_player_model = None
    viewmodel = None  # first-person arms glued to the camera
    weapon = None     # the local player's current weapon (Modules/Weapons)
    boom_arm = None
    third_person = False
    toggle_third_person = lambda key: None

    def setup_game(scene_key):
        nonlocal boom_arm, current_scene, current_scene_key, local_player_model, viewmodel, weapon, player, player_height, third_person, toggle_third_person
        current_scene_key = scene_key
        current_scene = get_or_load_scene(current_scene_key)

        # Player capsule: walks/collides against current_scene's static
        # geometry (see torus_scene.py's collision=True statics) via the
        # scene's own physics world. Spawned above the floor so it falls
        # and settles on first update rather than starting embedded in it.
        # max_slope_degrees raised slightly past Source's 45.57 default - the
        # TorusScene staircase's clip-brush ramp (see torus_scene.py) sits at
        # ~46.3 degrees, fit to the actual tread-nosing line rather than a
        # shallower approximation, so it needs a hair more headroom to count
        # as walkable floor instead of a wall.
        # CharacterController.get_eye_offset() (non-crouched, see
        # character_controller.py) puts the camera at height/2 +
        # height*_DEFAULT_EYE_RATIO above the feet, where _DEFAULT_EYE_RATIO
        # is 0.7/1.8 - i.e. eye level = height * 8/9. The rat.glb model's
        # actual measured eye level is 1.39225m above its feet, so height is
        # set here to whatever makes that formula land exactly there
        # (1.39225 * 9/8), rather than picking an arbitrary height and
        # letting the eye level fall wherever the generic ratio puts it -
        # keeps the first-person camera at the same height the visible/
        # shadow-casting model's own eyes actually are.
        player_height = 1.39225 * 9 / 8
        player = CharacterController(
            current_scene.physics,
            position=SPAWN_POSITION,
            height=player_height,
            max_slope_degrees=47.0,
        )

        # The local player's own visual body - shadow-only (visible_in_color
        # =False) since a first-person player never sees their own model,
        # only what it casts onto the ground. Model path/animation are
        # hardcoded HERE, at the call site, rather than inside PlayerModel
        # itself, so swapping the local player's visual model later is a
        # one-line change - PlayerModel (Modules/Player/player_model.py)
        # stays fully generic, not tied to this one asset.
        # All of this project's rat/rifle animations were baked with
        # Blender's factory-default scene frame rate (24fps) left unchanged,
        # even though they were actually authored/intended for 30fps - so
        # every raw keyframe time in these files is 30/24 = 1.25x too long
        # (each file plays back 25% slower than intended). RAT_ANIM_TIME_SCALE
        # corrects for that by rescaling every keyframe time by 24/30 before
        # it's used - see skeletal_loader.load_skinned_glb/
        # load_animation_clips' own time_scale docstring for the mechanism.
        RAT_ANIM_TIME_SCALE = 24.0 / 30.0

        # Blend-state table: (name, clip, min_speed), min_speed in m/s.
        # Source's own velwalk/velrun (90/220 Source units/s * 0.0254 - the
        # same constants character_controller.py's footstep-sound code uses)
        # decide when "walk"/"run" kick in. rat.glb's own baked-in "New" clip
        # is actually a dancing animation, not an idle one - "rifle_idle"/
        # "rifle_walk"/"rifle_run" (loaded below from the separate pose-only
        # files, sharing rat.glb's armature) are the real poses. All three
        # are full-body clips (legs included, not just arms), so the same
        # table is reused for the upper-body split below too - appending a
        # new blend state later (crouch-walk, sprint, whatever) is just
        # adding another triple here, PlayerModel needs no other change.
        player_animation_states = (
            ("idle", "rifle_idle", 0.0),
            ("walk", "rifle_walk", 90.0 * 0.0254),
            ("run", "rifle_run", 220.0 * 0.0254),
        )

        # Facing-relative directional overrides for the "walk" state only -
        # RifleWalkS/E/W/NE/NW/SE/SW.glb (loaded below) exist alongside
        # RifleWalkN.glb, but there's no equivalent run-direction set yet,
        # so "run" deliberately has no entry here at all and keeps playing
        # rifle_run regardless of direction until those are authored too -
        # see PlayerModel's directional_clips docstring for exactly how a
        # missing state/direction falls back to the plain (or nearest-
        # cardinal) clip.
        player_directional_clips = {
            "walk": {
                "N": "rifle_walk",
                "S": "rifle_walk_s",
                "E": "rifle_walk_e",
                "W": "rifle_walk_w",
                "NE": "rifle_walk_ne",
                "NW": "rifle_walk_nw",
                "SE": "rifle_walk_se",
                "SW": "rifle_walk_sw",
            },
        }

        local_player_model = PlayerModel(
            current_scene,
            "Assets/Models/rat.glb",
            visible_in_color=False,
            cast_shadow=True,
            time_scale=RAT_ANIM_TIME_SCALE,
            tint_mask_path=RAT_TINT_MASK_PATH,
            states=player_animation_states,
            directional_clips=player_directional_clips,
            # Splits the model into a lower body (locomotion - legs, spine,
            # neck, head - above) and an upper body (gun-holding pose)
            # driven independently. Deliberately just the two clavicles, NOT
            # their shared parent Spine4 - Spine4 also parents Neck1 (->
            # Head1), and Bip01_Spine/Spine1/Spine2 above IT drive the torso
            # twist - rooting the split there pulled the spine AND neck/head
            # into the "upper body" mask too, so a gun-holding pose (or the
            # shotgun-idle override) would visibly override the head/neck
            # look-direction and spine lean along with the arms, not just
            # the arms. Rooting at the clavicles instead confines the swap
            # to exactly the shoulder/arm/hand chains - spine, neck, and
            # head always follow the LOWER body's own locomotion clip
            # (idle/walk/run/jump/crouch) no matter what the upper body is
            # doing.
            upper_body_root_joints=[
                "ValveBiped.Bip01_R_Clavicle",
                "ValveBiped.Bip01_L_Clavicle",
            ],
            # A weapon's held pose (set_upper_override - see WeaponsBase.
            # equip_player) uses a WIDER split instead, rooted
            # at Spine4 - which also parents Neck1 (-> Head1) alongside both
            # clavicle chains in this rig, so this pulls in arms, neck, AND
            # head as one unit. Locomotion-driven upper-body poses above
            # stay narrower (clavicles only) on purpose, but an override
            # like pistol_idle was authored with its own Spine4/neck
            # rotation as PART of the pose - splitting it at the clavicles
            # instead fed that clip's clavicle rotation the wrong PARENT
            # transform (whatever Spine4 orientation the current locomotion
            # clip happened to be using), which is what actually made the
            # arm look rotated wrong. Widening the override's own mask to
            # include Spine4 lets it supply its own correct pose for
            # everything above the shoulders, matching how it was actually
            # authored - no manual per-joint rotation correction needed.
            override_upper_body_root_joints=["ValveBiped.Bip01_Spine4"],
            upper_states=player_animation_states,
            # RifleJump.glb (loaded below) - a one-shot takeoff pose, not a
            # speed-driven blend state like the three above, so it's its own
            # param rather than another `states` entry: PlayerModel.update()
            # overrides BOTH bodies with it the instant is_grounded reads
            # False, holding its last frame (see add_skeletal's own loop
            # param) until is_on_ground() is true again, at which point
            # normal idle/walk/run switching resumes.
            jump_animation="rifle_jump",
            # RifleCrouch.glb (loaded below) - a static held pose (2
            # identical keyframes, confirmed via raw glb inspection - not a
            # cycle), not a speed-driven blend state, so it's its own param
            # like jump_animation above rather than another `states` entry.
            # PlayerModel.update() overrides BOTH bodies with it for as long
            # as is_crouched (already passed below via player.is_crouched())
            # reads True, resuming normal idle/walk/run switching the
            # instant it reads False again.
            crouch_animation="rifle_crouch",
        )
        # Matches torus_scene.py's own decorative rat.glb character's
        # shading exactly (same flat "character["specular_strength"] = 0"
        # mutation there) - PlayerModel has no constructor knob for this
        # (bind_material's own default is 1.0, a normal specular highlight),
        # so it's set directly on the obj dict here, same as the decorative
        # one does.
        if local_player_model.obj is not None:
            local_player_model.obj["specular_strength"] = 0
        # First-person arms (see Modules/Player/viewmodel.py) - drawn only
        # while the camera is in first person, tinted with the same fur color.
        viewmodel = ViewModel(current_scene)
        # rifleidle.glb/RifleWalkN.glb are separate pose-only exports sharing
        # rat.glb's own armature (see load_additional_animations) - both
        # files happen to name their one clip "New" (same name rat.glb's own
        # base clip already uses), hence the rename to keep all three
        # distinct on the merged skeleton.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/rifleidle.glb",
            rename={"New": "rifle_idle"},
            time_scale=RAT_ANIM_TIME_SCALE,
        )
        # RifleWalkN.glb specifically has ALREADY been re-exported at a
        # correct 30fps (confirmed: its own keyframes are spaced at exactly
        # 1/30s, unlike rat.glb/rifleidle.glb above, still at 1/24s) - no
        # time_scale correction here, or this would double-correct an
        # already-fixed file and play it 20% too fast.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkN.glb",
            rename={"New": "rifle_walk"},
        )
        # RifleWalkS.glb/RifleWalkE.glb - re-exported since this was first
        # wired in and now BOTH already bake at a correct 30fps (confirmed
        # via raw glb inspection: 28 keyframes/0.9s and 23 keyframes/0.7333s
        # respectively, both exactly 30fps) - no time_scale correction, same
        # as RifleWalkN.glb.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkS.glb",
            rename={"New": "rifle_walk_s"},
        )
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkE.glb",
            rename={"New": "rifle_walk_e"},
        )
        # RifleWalkW.glb - unlike S/E above, still baked at 24fps (confirmed:
        # 18 keyframes over 0.7083s), so it still needs the correction or it
        # plays 25% slower than intended.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleWalkW.glb",
            rename={"New": "rifle_walk_w"},
            time_scale=RAT_ANIM_TIME_SCALE,
        )
        # RifleWalkNE/NW/SE/SW.glb - each of these bundles 34 clips (a full
        # animation library re-export, not just the one directional pose),
        # all byte-identical to each other across the 4 files EXCEPT the one
        # literally named "New" - confirmed by directly comparing every
        # clip's keyframe data across all 4 files. That's the same clip name
        # every other single-pose file here (rifleidle.glb, RifleWalkN.glb,
        # etc.) uses for its real authored pose, so it's used here too rather
        # than the more prominent-looking "Diag" clip, which turned out to be
        # byte-for-byte identical junk shared by all 4 files (a leftover
        # reference/placeholder track, not the actual walk motion). All 4
        # confirmed at 24fps via the same keyframe-spacing check as
        # RifleWalkW.glb above, so all 4 need the correction.
        for _direction, _filename in (
            ("ne", "RifleWalkNE.glb"),
            ("nw", "RifleWalkNW.glb"),
            ("se", "RifleWalkSE.glb"),
            ("sw", "RifleWalkSW.glb"),
        ):
            current_scene.load_additional_animations(
                local_player_model.obj,
                f"Assets/Animations/Poses/Rifle/{_filename}",
                rename={"New": f"rifle_walk_{_direction}"},
                time_scale=RAT_ANIM_TIME_SCALE,
            )
        # RifleRunN.glb - also already baked at a correct 30fps (confirmed
        # the same way as RifleWalkN.glb), no time_scale correction needed.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleRunN.glb",
            rename={"New": "rifle_run"},
        )
        # RifleJump.glb - confirmed already baked at a correct 30fps (same
        # raw-keyframe-spacing check as RifleWalkN.glb/RifleRunN.glb), no
        # time_scale correction needed.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleJump.glb",
            rename={"New": "rifle_jump"},
        )
        # RifleCrouch.glb - a static pose (its own first/last keyframe are
        # identical, confirmed via raw glb inspection), so its authored
        # frame rate/duration don't matter - no time_scale correction needed.
        current_scene.load_additional_animations(
            local_player_model.obj,
            "Assets/Animations/Poses/Rifle/RifleCrouch.glb",
            rename={"New": "rifle_crouch"},
        )
        # Third-person mode - see Modules/Camera/camera_boom_arm.py. Off by
        # default (first person): the local player's own model stays
        # shadow-only (visible_in_color=False above) until this is toggled.
        third_person = False
        boom_arm = CameraBoomArm(current_scene.physics)

        def toggle_third_person(key):
            nonlocal third_person
            if key == pygame.K_v:
                third_person = not third_person
                # The local player's body is normally shadow-only (a first-
                # person player never sees their own model - see
                # local_player_model's own visible_in_color=False above) -
                # third person needs it actually drawn instead.
                local_player_model.set_visible_in_color(third_person)

        # The current weapon decides everything about how the player holds it:
        # its gun and arm animations on the first-person viewmodel, its world
        # model in the character's hand, and the pose of the player rig's upper
        # body (plus that pose's rotation corrections - see WeaponsBase).
        weapon = USP()
        weapon.equip_viewmodel(viewmodel)
        weapon.equip_player(current_scene, local_player_model)

    ui = UIManager(window)
    ui_demo = build_demo(ui)

    # TEMPORARY health bar - a placeholder until a real damage system exists.
    # H takes 10 damage, J heals 10 (see on_key_down); nothing else reads or
    # writes `health` yet. Hidden on the main menu, shown once a map starts.
    name_tags = NameTags(ui)
    scoreboard = Scoreboard(ui)
    health ={"value": 100.0, "max": 100.0}
    health_bar = ProgressBar(
        value=health["value"], max_value=health["max"], label_format="{value:.0f} / {max:.0f}",
        anchor=Anchor.BOTTOM_LEFT, offset=(130, -49), size=(300, 26), visible=False,
        fill_color=(90, 200, 110, 255),
    )
    ui.root.add(health_bar)
    # Screen-centre crosshair; its gap is the current weapon's real spread.
    crosshair = ui.root.add(Crosshair(visible=False))
    # Diagonal ticks over the crosshair when one of our shots hits another player.
    hitmarker = ui.root.add(Hitmarker())
    hit_sound_channel = [None]   # every hit plays on the same channel, cutting off the last

    # Dying: health reaching 0 (from any source) flags `pending`; the main loop starts
    # the death at a safe point (begin_death) and ends it (end_death) when the
    # screen's countdown runs out. `eye_drop` eases the camera down to the floor.
    death = {"pending": False, "active": False, "eye_drop": 0.0}
    death_screen = ui.root.add(DeathScreen(RESPAWN_SECONDS))

    def change_health(delta):
        if death["active"] and delta < 0.0:
            return   # already dead
        health["value"] = max(0.0, min(health["max"], health["value"] + delta))
        health_bar.value = health["value"]
        low = health["value"] <= health["max"] * 0.3
        health_bar.fill_color = (220, 70, 70, 255) if low else (90, 200, 110, 255)
        if health["value"] <= 0.0 and not death["active"]:
            death["pending"] = True

    # The game starts on the main menu and loads nothing else. Host/Join
    # (see Modules/UI/lobby_menu.py) talk to NetworkManager - created here,
    # before any map exists, since the menu needs Steam callbacks pumped -
    # and once a lobby is hosted/joined the menu reports the chosen map via
    # request_start. The real work (setup_game: map, player, models, each
    # behind its own loading frame) runs from the main loop, not inside
    # that callback, which can fire from inside Steam's callback dispatch.
    MAPS = [("Main Map", "mainmap"), ("Torus (test)", "torus")]
    in_menu = True
    pending_start = None
    net_mgr = NetworkManager(camera, None)
    tracers = Tracers(window.ctx)
    particles = ParticleManager(window.ctx, material_aliases=MATERIAL_ALIASES,
                                radius_scale=RADIUS_SCALE, frame_filter=keep_frame,
                                lifetime_scale=LIFETIME_SCALE)
    particles.load("Assets/Particles/Muzzle/muzzleflashes.pcf")
    particles.load(IMPACT_FILE)
    register_blood(particles)
    # Particles that collide (impact debris) trace against the level only.
    particles.raycast = lambda a, b: (
        current_scene.physics.raycast(a, b, CollisionGroup.STATIC) if current_scene is not None else None)

    def remote_shot(start, end, follow=None):
        """Another player's shot: its tracer, a muzzle flash that stays on their gun, and
        the impact where it ended (looked up here - only the end point is sent)."""
        tracers.add(start, end)
        aim = glm.vec3(end) - glm.vec3(start)
        if glm.length(aim) > 1e-3:
            if current_scene is not None:
                spawn_impact(particles, current_scene.physics.raycast(
                    start, glm.vec3(end) + glm.normalize(aim) * 0.1))
            particles.spawn(WeaponsBase.muzzle_particle, start, forward=aim,
                            colors=WeaponsBase.muzzle_color, size=WeaponsBase.muzzle_size,
                            offset_scale=WeaponsBase.muzzle_offset_scale, follow=follow)
    net_mgr.on_tracer = remote_shot

    def remote_death(position, velocity, color=None):
        """Another player died: their body bursts into gibs (in their fur colour) where they stood."""
        if current_scene is not None and current_scene.gibs is not None:
            current_scene.gibs.spawn(position, velocity, tint=color)
    net_mgr.on_death = remote_death
    # Another player's shot hit us: the shooter decided that, we apply it.
    net_mgr.on_damage = lambda amount, attacker_id, weapon_name: change_health(-amount)
    from Modules.Scenes import scene_base as _scene_base
    _scene_base.LOAD_PUMP = net_mgr.pump_callbacks
    menu_scene = get_or_load_scene("mainmenu")
    game_look = (camera.yaw, camera.pitch, glm.vec3(camera.front))

    def request_start(map_key):
        nonlocal pending_start
        pending_start = map_key

    paper_doll = None  # head of the local rat beside the health bar - built in start_game

    def start_game(map_key):
        nonlocal in_menu, paper_doll
        menu_ui.visible = False
        setup_game(map_key)
        health_bar.visible = True
        crosshair.visible = True
        name_tags.visible = True
        if local_player_model is not None and local_player_model.obj is not None:
            local_player_model.set_hat(net_mgr.local_hat)
            local_player_model.set_tint(net_mgr.local_color)
            viewmodel.set_tint(net_mgr.local_color)
            paper_doll = PaperDoll(window.ctx, local_player_model.obj)
        prime_effects()
        in_menu = False
        ui.set_cursor_free(False)
        camera.yaw, camera.pitch, camera.front = game_look
        window.reset_frame_timer()
        net_mgr.scene = current_scene
        net_mgr.in_game = True

    def set_local_body_visible(visible):
        """Shows or hides the local player's body (and gun) - hidden while dead."""
        for obj in (local_player_model.obj, getattr(weapon, "worldmodel_obj", None)):
            if obj is not None:
                obj["visible_in_color"] = visible and third_person
                obj["cast_shadow"] = visible

    def begin_death():
        death["pending"] = False
        death["active"] = True
        death["eye_drop"] = 0.0
        feet = player.get_position() - glm.vec3(0.0, player_height / 2.0, 0.0)
        if current_scene.gibs is not None:
            current_scene.gibs.spawn(feet, player.velocity, tint=net_mgr.local_color)
        net_mgr.notify_death()
        set_local_body_visible(False)
        crosshair.visible = False
        player.set_move_direction(glm.vec3(0.0))
        player.set_sprinting(False)
        death_screen.show()

    def end_death():
        death["active"] = False
        player.teleport(SPAWN_POSITION)
        change_health(health["max"])
        if weapon is not None:
            weapon.recoil.reset()
        set_local_body_visible(True)
        crosshair.visible = True
        death_screen.hide()
        net_mgr.notify_respawn()

    def prime_effects():
        """Warms everything a first death, shot or hit would load or compile lazily - gib
        meshes and materials, blood and impact textures, sounds, the hit marker and death screen
        - so none of it hitches mid-game. All drawn into the back buffer only."""
        show_loading_screen(window, "Loading effects...")
        eye = glm.vec3(*SPAWN_POSITION) + glm.vec3(0.0, 0.5, 0.0)
        prime_camera = Camera(position=eye, aspect=camera.aspect)
        prime_camera.fov = camera.fov
        prime_camera.yaw, prime_camera.pitch = -90.0, 0.0
        prime_camera.update_vectors()
        sounds = current_scene.sound_manager
        if weapon is not None and weapon.fire_sound:
            sounds.preload(weapon.fire_sound, muffle=True)
        sounds.preload(HITMARKER_SOUND, gain=HITMARKER_GAIN)
        sounds.preload(HITMARKER_SOUND, gain=HITMARKER_GAIN, pitch=HITMARKER_HEADSHOT_PITCH)
        if current_scene.gibs is not None:
            if local_player_model is not None and local_player_model.obj is not None:
                current_scene.gibs.match_material(local_player_model.obj)
            current_scene.gibs.prime(prime_camera, tint=net_mgr.local_color)
        particles.prime(prime_camera)
        hitmarker.prime()
        death_screen.show()       # one invisible frame builds its text
        ui.render()
        death_screen.hide()
        window.reset_frame_timer()

    menu_ui = menu_scene.build_ui(ui, net_mgr, MAPS, on_start=request_start)
    ui.set_cursor_free(True)

    def toggle_ui_demo(key):
        if key == pygame.K_F1:
            ui_demo.visible = not ui_demo.visible
            ui.set_cursor_free(ui_demo.visible)

    trigger_clicks = [0]   # left clicks since the last frame's firing check

    def count_click(event):
        """window.handle_events' event filter: counts left clicks (from the
        event queue, so none is lost between frames) and then lets the UI have
        the event as before."""
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and not in_menu and not ui.cursor_free):
            trigger_clicks[0] += 1
        return ui.handle_event(event)

    def on_key_down(key):
        if in_menu:
            return
        if death["active"]:
            return
        toggle_third_person(key)
        toggle_ui_demo(key)
        if key == pygame.K_h:
            change_health(-10.0)
        elif key == pygame.K_j:
            change_health(10.0)

    running = True
    while running:
        running, dt = window.handle_events(
            camera, on_key_down=on_key_down, event_filter=count_click
        )

        if pending_start is not None:
            start_map, pending_start = pending_start, None
            start_game(start_map)

        if in_menu:
            net_mgr.update()
            menu_scene.update(dt)
            menu_scene.apply_camera(camera)
            window.ctx.clear(0.1, 0.1, 0.1, 1.0)
            menu_scene.render(camera, None)
            ui.render()
            window.flip()
            continue

        # Handle scene switching inputs (1: torus test scene, 2: mainmap)
        keys = pygame.key.get_pressed()
        new_scene_key = None
        if keys[pygame.K_1]:
            new_scene_key = "torus"
        if keys[pygame.K_2]:
            new_scene_key = "mainmap"
        # Gated on an ACTUAL change, not just "key held" - keys[...] is
        # a per-frame snapshot (true every frame the key stays down, not
        # just the one it was first pressed), and pause_all/resume_all
        # below don't need to run every single one of those frames.
        if new_scene_key is not None and new_scene_key != current_scene_key:
            # See SoundManager.pause_all's own docstring - every Scene
            # keeps running/existing once switched away from, including
            # any looping ambient sound it started, unless explicitly
            # paused here.
            current_scene.sound_manager.pause_all()
            current_scene_key = new_scene_key
            current_scene = get_or_load_scene(current_scene_key)
            current_scene.sound_manager.resume_all()

        # Ground-relative movement, driven by camera yaw (mouse-look)
        # but ignoring pitch - walking shouldn't speed up/slow down
        # just from looking up or down.
        if death["pending"] and not death["active"]:
            begin_death()
        alive = not death["active"]
        move_dir = glm.vec3(0.0)
        if keys[pygame.K_w]:
            move_dir += camera.get_flat_forward()
        if keys[pygame.K_s]:
            move_dir -= camera.get_flat_forward()
        if keys[pygame.K_a]:
            move_dir -= camera.get_flat_right()
        if keys[pygame.K_d]:
            move_dir += camera.get_flat_right()
        if alive:
            player.set_move_direction(move_dir)
            player.set_sprinting(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
            player.set_crouching(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL])
            if keys[pygame.K_SPACE]:
                player.jump()
        # No player.update(dt) here - CharacterController's Source-style
        # ground/air movement math runs once per fixed physics tick (a
        # PhysicsWorld pre-substep callback, registered in its __init__),
        # not once per variable-length render frame; current_scene.update()
        # below is what actually advances physics.

        # Call update to drive scene animations (like the rotating
        # torus) and step physics - camera position then follows
        # wherever physics moved the player capsule to this frame.
        current_scene.update(dt)
        if current_scene.gibs is not None:
            current_scene.gibs.update(dt)     # before particles.update: the blood follows the gibs
        eye_position = player.get_position() + glm.vec3(
            0.0, player.get_eye_offset(), 0.0
        )
        if not alive:
            # The view sinks to the floor over half a second, like the body dropping.
            death["eye_drop"] = min(1.0, death["eye_drop"] + dt / 0.5)
            ease = death["eye_drop"] * death["eye_drop"] * (3.0 - 2.0 * death["eye_drop"])
            eye_position.y -= (player.get_eye_offset() - 0.25) * ease
        if third_person:
            # Same pivot first-person already uses (the player's own eye
            # position) - camera.front/yaw/pitch (mouse look) and all
            # movement math are completely unaffected by this branch,
            # only WHERE the camera itself sits changes.
            camera.position = boom_arm.get_camera_position(
                eye_position, camera.yaw, camera.pitch
            )
        else:
            camera.position = eye_position
        # After the camera has its final position/look for this frame.
        # Recoil moves the camera a little further each frame (see Recoil) -
        # after mouse look, before anything reads camera.front for this frame.
        if weapon is not None and alive:
            weapon.recoil.apply(camera, dt)
        viewmodel.update(camera, not third_person and alive)

        # Left mouse fires: one shot per click (every click counts, even two
        # inside one frame), or - for an automatic weapon - continuously while
        # held, limited by the weapon's own fire_interval. Clicks are only
        # counted with the cursor captured (see count_click), so clicking
        # menus/UI never shoots.
        clicks, trigger_clicks[0] = trigger_clicks[0], 0
        if weapon is not None:
            crosshair.set_spread(weapon.spread_degrees(), camera.fov)
        if weapon is not None and not ui.cursor_free and alive:
            if weapon.automatic:
                clicks = 1 if pygame.mouse.get_pressed()[0] else 0
            for _ in range(clicks):
                # A line trace from the camera along the aim (see PhysicsWorld.
                # raycast): the first thing it meets - wall, prop or another
                # player's hitbox - is what the shot hit.
                shot = weapon.fire(
                    current_scene, camera.position, direction=camera.front,
                    follow=lambda: camera.position,
                )
                if shot is not None:
                    end = (shot.hit.position if shot.hit is not None
                           else camera.position + shot.direction * weapon.max_range)
                    net_mgr.notify_shot(glm.vec3(end))
                    # From the gun's muzzle bone to where the trace ended.
                    muzzle = weapon.muzzle_position(current_scene)
                    if muzzle is None:   # no gun model posed yet: from just off the eye
                        right = glm.normalize(glm.cross(camera.front, camera.up))
                        muzzle = camera.position + camera.front * 0.6 + right * 0.14 - camera.up * 0.12
                    tracers.add(muzzle, end)
                    spawn_impact(particles, shot.hit)
                    if weapon.muzzle_particle:
                        # overlay while the first-person gun is what's on screen (its depth is squashed)
                        particles.spawn(weapon.muzzle_particle, muzzle, forward=shot.direction,
                                        up=camera.up, overlay=not third_person,
                                        colors=weapon.muzzle_color, size=weapon.muzzle_size,
                                        offset_scale=weapon.muzzle_offset_scale,
                                        follow=lambda: weapon.muzzle_position(current_scene))
                    if shot.victim in net_mgr.remote_players:
                        net_mgr.send_damage(shot.victim, shot.damage, weapon.name)
                        hitmarker.trigger(headshot=shot.headshot)
                        # Flat, in both ears, at any distance: it's feedback for us, not a sound in the world.
                        hit_emitter = current_scene.sound_manager.add_sound(
                            HITMARKER_SOUND, camera.position, volume=HITMARKER_VOLUME, loop=False,
                            universal=True, channel=hit_sound_channel[0], gain=HITMARKER_GAIN,
                            pitch=HITMARKER_HEADSHOT_PITCH if shot.headshot else 1.0)
                        hit_sound_channel[0] = hit_emitter["channel"]

        # get_position() is the hull CENTER, not feet - subtract half
        # the standing height (the same player_height passed to
        # CharacterController above, not a re-read of any private/
        # crouch-varying internal) to get where the model should
        # actually stand. horiz_speed matches the flat/horizontal
        # convention already used elsewhere in this file (get_flat_
        # forward, footstep gating) so vertical jump/fall speed never
        # triggers a "running" pose.
        feet_position = player.get_position() - glm.vec3(0.0, player_height / 2.0, 0.0)
        horiz_speed = glm.length(glm.vec3(player.velocity.x, 0.0, player.velocity.z))
        # A one-shot consumed event (see CharacterController.pop_jumped's
        # own docstring, mirroring pop_footstep below) - popped BEFORE
        # the update() call it feeds, so PlayerModel can play the jump
        # takeoff pose the instant a jump actually executes instead of
        # waiting on is_on_ground()'s own debounced reading (see
        # PlayerModel.update()'s own just_jumped docstring for why that
        # debounce alone made a real jump feel delayed).
        just_jumped = player.pop_jumped()
        if weapon is not None:
            weapon.set_moving(horiz_speed)
        local_player_model.update(
            dt,
            feet_position,
            camera.yaw,
            horiz_speed,
            is_crouched=player.is_crouched(),
            is_grounded=player.is_on_ground(),
            # Walk-vs-run is now driven by the actual sprint key input,
            # not momentum - see PlayerModel.update()'s own is_sprinting
            # docstring for why (this was the real fix for the run
            # animation "randomly restarting", which turned out to be
            # physics speed noise crossing a threshold, not an animation
            # data or looping-code problem).
            is_sprinting=player.is_sprinting(),
            # Same raw WASD-derived direction vector already passed to
            # player.set_move_direction() above - drives
            # player_directional_clips' facing-relative N/S/E/W walk
            # selection (see PlayerModel.update()'s own move_direction
            # docstring). Reused as-is rather than reading it back off
            # CharacterController, since that's exactly the vector this
            # was built from a few lines up.
            move_direction=move_dir,
            just_jumped=just_jumped,
        )
        # The same values driving the local model, sent to other players so
        # their copy of us moves and animates identically (see NetworkManager.
        # set_local_state for why this isn't just the camera position).
        net_mgr.set_local_state(
            feet_position, camera.yaw, camera.pitch, horiz_speed,
            player.is_crouched(), player.is_on_ground(), player.is_sprinting(),
            move_dir, just_jumped,
        )

        footstep = player.pop_footstep()
        if footstep is not None:
            material, volume = footstep
            current_scene.play_footstep_sound(
                material, player.get_position(), volume=volume
            )

        current_scene.update_audio(camera)

        if death["active"]:
            death_screen.update()
            if death_screen.finished:
                end_death()

        net_mgr.update()
        name_tags.update(net_mgr.remote_players, camera, window.ctx.screen.size)
        scoreboard.visible = bool(keys[pygame.K_TAB])
        if scoreboard.visible:
            scoreboard.update(net_mgr)

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        # First-person muzzle flashes (overlay effects) are drawn by the scene just
        # before the viewmodels, so the gun and arms sit in front of them.
        current_scene.before_viewmodels = lambda: particles.render(camera, overlay=True)
        current_scene.render(camera, None)
        tracers.render(camera)
        particles.update(dt)
        particles.render(camera, overlay=False)
        ui.render()
        if paper_doll is not None:
            paper_doll.render(window.ctx.screen.size)
        window.flip()

    particles.destroy()
    tracers.destroy()
    ui.destroy()
    window.quit()
    sys.exit()


def _write_crash_log(exc):
    """Appends a full traceback (plus a few basics: frozen or not,
    platform) to crash_log.txt next to the running exe (or the project
    root, for a dev `python app.py` run - see _is_frozen_build's own
    docstring for that distinction). Exists specifically because
    BuildCMD_Nuitka builds with --windows-console-mode=disable - a
    frozen build has NO console for an unhandled exception's traceback
    to print to at all, so today an unhandled crash (a GL context that
    fails to create because a given GPU/driver doesn't support some
    requested feature is the leading suspect for a reported AMD-only
    crash - see the entry point below) is completely silent: the window
    just flashes and closes, with nothing anywhere explaining why. This
    doesn't fix any particular crash - it exists so the NEXT one leaves
    behind something to actually diagnose it from, rather than staying
    a guess. Never lets a failure IN this logging itself replace the
    original exception - see the entry point's own re-raise.

    Deliberately NOT written "next to the exe" via sys.executable - a
    first version of this did exactly that, and it was WRONG for a
    --onefile build specifically: Nuitka's own onefile bootstrap
    unpacks the real program into a TEMPORARY directory and runs it
    from there (confirmed directly in Nuitka's own C source,
    OnefileBootstrap.c) - sys.executable inside that process points
    INTO that temp directory, which the bootstrap deletes the moment
    this process exits, crash or not. A crash log written there was
    already gone by the time anyone went looking for it - it looked
    exactly like no log had been written at all. ~/.ratwar isn't
    touched by that cleanup and survives the process exiting, which is
    the entire point of a crash log."""
    import traceback
    import datetime
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".ratwar")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "crash_log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== crash at {datetime.datetime.now().isoformat()} ===\n")
            f.write(f"frozen build: {_is_frozen_build()}\n")
            f.write(f"platform: {sys.platform}\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _write_crash_log(e)
        raise
