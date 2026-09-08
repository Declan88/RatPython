import sys
import pygame
import os
import ctypes
from Modules.Window.window import WindowManager
from Modules.Camera.camera import Camera
from Modules.Scenes.torus_scene import TorusScene


def load_steam_api_dll():
    """Finds and pre-loads steam_api64.dll globally (RTLD_GLOBAL) before
    any module imports py_steam_net, which links against it. No
    hardcoded machine-specific paths - just sensible, portable
    candidate locations, plus an environment variable escape hatch for
    anything unusual (a packaged build laying it out differently, etc.)."""
    if sys.platform != "win32":
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(script_dir, "steam_api64.dll"),
        os.path.join(os.getcwd(), "steam_api64.dll"),
    ]

    env_override = os.environ.get("STEAM_API_DLL_PATH")
    if env_override:
        candidates.insert(0, env_override)

    for path in candidates:
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                os.add_dll_directory(os.path.dirname(path))
                print(f"Pre-loaded steam_api64.dll from: {path}")
                return
            except Exception as e:
                print(f"Found steam_api64.dll at {path} but failed to load it: {e}")

    print(
        "Warning: steam_api64.dll not found next to app.py or in the "
        "current working directory. Networking will likely fail to "
        "import py_steam_net. Set the STEAM_API_DLL_PATH environment "
        "variable to point at it directly if it's somewhere else."
    )


# Pre-load steam_api64.dll globally before any modules import py_steam_net
load_steam_api_dll()

from Modules.Networking.network_manager import NetworkManager


def main():
    print("Yo wsg")
    window = WindowManager(800, 600, "Pygame-ce Engine - Scene Switcher")
    camera = Camera(position=(0.0, 0.0, 3.0), aspect=window.width / window.height)

    # Load initial scenes dictionary
    scenes = {"torus": TorusScene(window.ctx)}
    current_scene_key = "torus"
    current_scene = scenes[current_scene_key]

    net_mgr = NetworkManager(camera)

    running = True
    while running:
        running, dt = window.handle_events(camera)

        # Handle scene switching inputs (1: Triangle, 2: Cube, 3: Torus)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]:
            current_scene_key = "torus"
            current_scene = scenes[current_scene_key]

        camera.process_keyboard(keys, dt)

        # Call update to drive scene animations (like the rotating torus)
        current_scene.update(dt)
        current_scene.update_audio(camera)

        net_mgr.update()

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        current_scene.render(camera, None)
        window.flip()

    window.quit()
    sys.exit()


if __name__ == "__main__":
    main()
