import time
import pygame
import moderngl

class WindowManager:
    def __init__(self, width=800, height=600, title="3D Game"):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        self.width = width
        self.height = height
        self.title = title
        self.flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE

        # vsync=1 caps to the display's refresh rate; vsync=0 (toggled
        # at runtime via F10, see handle_events) uncaps it - pygame only
        # applies a vsync change through a fresh set_mode() call, same
        # as the existing VIDEORESIZE handling below.
        self.vsync = 1
        self.screen = pygame.display.set_mode((self.width, self.height), self.flags, vsync=self.vsync)
        pygame.display.set_caption(title)

        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        # Grabbing input warps the OS cursor to the window center, which
        # SDL can report back as a large synthetic MOUSEMOTION event (as
        # if the mouse had been yanked) - without this, that one event
        # feeds straight into Camera.process_mouse() as a real look
        # input, snapping pitch/yaw hard on the very first frame.
        # Clearing it here (and resetting the relative-motion
        # accumulator via get_rel()) discards it before the game loop
        # ever sees it.
        pygame.event.clear(pygame.MOUSEMOTION)
        pygame.mouse.get_rel()

        # No moderngl.MULTISAMPLE flag exists - GL_MULTISAMPLE is enabled
        # by default in a core profile context once the framebuffer
        # actually has multisample buffers (requested above via
        # GL_MULTISAMPLEBUFFERS/GL_MULTISAMPLESAMPLES), so no explicit
        # enable() call is needed or possible here.
        self.ctx = moderngl.create_context()
        self.clock = pygame.time.Clock()

        # Throttles how often the FPS readout in the title bar updates -
        # doing it every single frame is wasteful and the number would
        # be too jittery to read anyway.
        self._fps_display_timer = 0.0

        # pygame.time.Clock.tick() returns whole MILLISECONDS as an int
        # - fine at ~60fps (a 1ms rounding error is a small fraction of
        # a ~16.7ms frame) but useless as a dt source once vsync is off
        # and frames take under 1ms each: tick() then returns 0 or 1
        # almost every call, so movement speed (position += speed * dt)
        # ends up scaled by that quantization noise instead of real
        # elapsed time - exactly why toggling vsync visibly changed
        # player speed. dt is measured with perf_counter() instead
        # (float seconds, way finer than 1ms); self.clock.tick() is
        # still called every frame purely to keep get_fps() (the title
        # bar readout) working, its return value is otherwise unused.
        self._last_time = time.perf_counter()

    def toggle_vsync(self):
        self.vsync = 0 if self.vsync else 1
        self.screen = pygame.display.set_mode((self.width, self.height), self.flags, vsync=self.vsync)

    def handle_events(self, camera):
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        self.clock.tick()  # advances pygame's internal fps averaging only - see __init__'s comment

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, dt
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, dt
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                elif event.key == pygame.K_F10:
                    self.toggle_vsync()
            elif event.type == pygame.MOUSEMOTION:
                camera.process_mouse(event.rel[0], event.rel[1])
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), self.flags, vsync=self.vsync)
                self.ctx.viewport = (0, 0, self.width, self.height)
                camera.aspect = self.width / self.height

        self._fps_display_timer += dt
        if self._fps_display_timer >= 0.5:
            self._fps_display_timer = 0.0
            vsync_label = "vsync on" if self.vsync else "vsync off (uncapped)"
            pygame.display.set_caption(
                f"{self.title} - {self.clock.get_fps():.0f} FPS - {vsync_label}"
            )

        return True, dt

    def flip(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()