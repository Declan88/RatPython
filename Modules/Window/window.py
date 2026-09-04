import pygame
import moderngl

class WindowManager:
    def __init__(self, width=800, height=600, title="3D Game"):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        self.width = width
        self.height = height
        self.flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.width, self.height), self.flags, vsync=1)
        pygame.display.set_caption(title)

        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        self.ctx = moderngl.create_context()
        self.clock = pygame.time.Clock()

    def handle_events(self, camera):
        dt = self.clock.tick() / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, dt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, dt
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
            elif event.type == pygame.MOUSEMOTION:
                camera.process_mouse(event.rel[0], event.rel[1])
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), self.flags, vsync=1)
                self.ctx.viewport = (0, 0, self.width, self.height)
                camera.aspect = self.width / self.height

        return True, dt

    def flip(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()