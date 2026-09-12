import sys
import pygame
import os
import ctypes
import glm


def _is_frozen_build():
    """True for a PyInstaller build (sys.frozen) or a Nuitka-compiled
    one ("__compiled__" injected into this entry module's globals -
    Nuitka's own documented detection method, since it never sets
    sys.frozen at all) - False for an ordinary `python app.py` dev
    run either way. Shared by every frozen-only check in this file
    (asset-path chdir, GPU-preference registry key) so they can't
    silently drift out of sync with each other."""
    return (getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")) or "__compiled__" in globals()


def _chdir_for_frozen_build():
    """PyInstaller's --onefile mode extracts --add-data bundles (this
    project's Assets folder included) to a temporary directory at
    startup, exposed as sys._MEIPASS - NOT the working directory the
    exe was launched from, and NOT the exe's own folder either. Every
    asset path in this codebase (torus_scene.py's model/texture/audio
    paths, etc.) is a plain relative string like "Assets/Models/...",
    which only resolves correctly if the process's current working
    directory happens to BE that extraction folder. Since none of
    those call sites can be reached before this runs (they're all
    behind imports/calls below), chdir'ing here once, before anything
    else executes, makes every existing relative path keep working
    unmodified instead of touching dozens of individual asset-loading
    call sites. Below Assets: this exists only for the lifetime of the
    process, so anything that writes there at runtime (e.g.
    bake_static_lighting's lightmap cache) won't persist between runs
    of a --onefile exe - that's a separate, real limitation of
    --onefile for a cache that's meant to survive across runs, not
    something this chdir fixes or is trying to fix."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller (see BuildCMD).
        os.chdir(sys._MEIPASS)
    elif "__compiled__" in globals():
        # Nuitka - both --standalone and --onefile. Unlike PyInstaller,
        # Nuitka has no separate "_MEIPASS-equivalent" to look up: a
        # --onefile build re-executes itself from its own unpacked temp
        # directory before any of this code runs, so sys.executable
        # already points at the unpacked location in EITHER mode -
        # os.path.dirname(sys.executable) is the one lookup that works
        # for both. "__compiled__" is a module-global Nuitka injects
        # into the entry script specifically (not into sys) - this is
        # Nuitka's own documented way to detect a compiled run.
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
    moderngl/pygame-ce rendering does) - see _load_gpu_hint_dll below
    for the mechanism that actually works for OpenGL. Kept here anyway
    as a harmless belt-and-suspenders extra: costs nothing, and covers
    any future D3D-based rendering path or driver version where it
    does get honored.

    Keyed by the exact exe PATH (sys.executable - still the actual
    launched exe even under --onefile, not the temporary _MEIPASS
    extraction folder), so this only ever affects this one built exe,
    never other unrelated programs. Only runs for a frozen/built exe,
    not a dev `python app.py` run - a dev run's sys.executable is
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
            0, winreg.KEY_SET_VALUE,
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
# confirmed (see build_tools/pyinstaller_bootloader/README.md) that
# these MUST be in the actual .exe's own PE export table specifically;
# a DLL loaded at runtime does nothing, regardless of how early it's
# loaded here - an earlier version of this file tried exactly that
# (a companion gpu_hint.dll) and it had no effect. Since PyInstaller's
# stock bootloader is what becomes app.exe and has no exports of its
# own, the actual fix lives one level down the toolchain: a patched
# bootloader with those two symbols added to its source and rebuilt,
# installed into the local PyInstaller package - see
# build_tools/pyinstaller_bootloader/README.md for what was changed
# and how to reproduce it (e.g. after reinstalling/upgrading
# PyInstaller, which would silently restore the stock, unpatched
# bootloader). Nothing in this Python file can express that fix -
# there is no per-build-invocation flag or Python-level hook for it.

from Modules.Window.window import WindowManager
from Modules.Camera.camera import Camera
from Modules.Scenes.torus_scene import TorusScene
from Modules.Physics.character_controller import CharacterController
from Modules.Player.player_model import PlayerModel


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
    }.get(sys.platform, "libsteam_api.so")  # covers "linux" and any other POSIX platform

    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []

    # Check PyInstaller's temporary extraction folder first if frozen -
    # Nuitka needs no equivalent special-case here, since
    # _chdir_for_frozen_build() (called at module import time, above
    # this function's own call further down) already puts the process's
    # cwd at the right unpacked location for a Nuitka build on every
    # platform, which the os.getcwd() candidate below already covers.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(os.path.join(sys._MEIPASS, lib_name))

    candidates.extend(
        [
            os.path.join(script_dir, lib_name),
            os.path.join(os.getcwd(), lib_name),
        ]
    )

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

    # Load initial scenes dictionary
    scenes = {"torus": TorusScene(window.ctx)}
    current_scene_key = "torus"
    current_scene = scenes[current_scene_key]

    # Player capsule: walks/collides against current_scene's static
    # geometry (see torus_scene.py's collision=True statics) via the
    # scene's own physics world. Spawned above the floor so it falls
    # and settles on first update rather than starting embedded in it.
    # max_slope_degrees raised slightly past Source's 45.57 default - the
    # TorusScene staircase's clip-brush ramp (see torus_scene.py) sits at
    # ~46.3 degrees, fit to the actual tread-nosing line rather than a
    # shallower approximation, so it needs a hair more headroom to count
    # as walkable floor instead of a wall.
    player_height = 1.5
    player = CharacterController(
        current_scene.physics,
        position=(0.0, 2.0, 3.0),
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
    local_player_model = PlayerModel(
        current_scene, "Assets/Models/rat.glb",
        visible_in_color=False, cast_shadow=True,
        idle_animation="funnyrat_ARMAction",
    )

    net_mgr = NetworkManager(camera, current_scene)

    running = True
    while running:
        running, dt = window.handle_events(camera)

        # Handle scene switching inputs (1: Triangle, 2: Cube, 3: Torus)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            current_scene_key = "torus"
            current_scene = scenes[current_scene_key]

        # Ground-relative movement, driven by camera yaw (mouse-look)
        # but ignoring pitch - walking shouldn't speed up/slow down
        # just from looking up or down.
        move_dir = glm.vec3(0.0)
        if keys[pygame.K_w]:
            move_dir += camera.get_flat_forward()
        if keys[pygame.K_s]:
            move_dir -= camera.get_flat_forward()
        if keys[pygame.K_a]:
            move_dir -= camera.get_flat_right()
        if keys[pygame.K_d]:
            move_dir += camera.get_flat_right()
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
        camera.position = player.get_position() + glm.vec3(
            0.0, player.get_eye_offset(), 0.0
        )

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
        local_player_model.update(dt, feet_position, camera.yaw, horiz_speed, is_crouched=player.is_crouched())

        footstep = player.pop_footstep()
        if footstep is not None:
            material, volume = footstep
            current_scene.play_footstep_sound(material, player.get_position(), volume=volume)

        current_scene.update_audio(camera)

        net_mgr.update()

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        current_scene.render(camera, None)
        window.flip()

    window.quit()
    sys.exit()


if __name__ == "__main__":
    main()
