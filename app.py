import sys
import pygame
from Modules.Window.window import WindowManager
from Modules.Camera.camera import Camera
from Modules.Scenes.torus_scene import TorusScene

def main():
    window = WindowManager(800, 600, "Pygame-ce Engine - Scene Switcher")
    camera = Camera(position=(0.0, 0.0, 3.0), aspect=window.width / window.height)

    # Load initial scenes dictionary
    scenes = {
        "torus": TorusScene(window.ctx)
    }
    current_scene_key = "torus"
    current_scene = scenes[current_scene_key]

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

        window.ctx.clear(0.1, 0.1, 0.1, 1.0)
        current_scene.render(camera, None)
        window.flip()

    window.quit()
    sys.exit()

if __name__ == "__main__":
    main()