import sys
import pygame
import os
import ctypes
from Modules.Window.window import WindowManager
from Modules.Camera.camera import Camera
from Modules.Scenes.torus_scene import TorusScene

# Pre-load steam_api64.dll globally before any modules import py_steam_net
if sys.platform == "win32":
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steam_api64.dll")
    if os.path.exists(dll_path):
        ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
        os.add_dll_directory(os.path.dirname(dll_path))

from Modules.Networking.network_manager import NetworkManager


def main():
    print("Welcome!")
    window = WindowManager(800, 600, "Pygame-ce Engine - Scene Switcher")
    camera = Camera(position=(0.0, 0.0, 3.0), aspect=window.width / window.height)

    # Load initial scenes dictionary
    scenes = {
        "torus": TorusScene(window.ctx)
    }
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

        net_mgr.update()

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        current_scene.render(camera, None)
        window.flip()

    window.quit()
    sys.exit()

if __name__ == "__main__":
    main()