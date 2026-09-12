import os
import sys
import time
import pygame
import moderngl


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if sys.platform == "win32":
    import ctypes

    myappid = "mycompany.mygame.subproduct.version"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    _SM_CXSMICON = 49
    _SM_CYSMICON = 50
    _ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
    _WM_SETICON = 0x0080
    _ICON_SMALL = 0
    _ICON_BIG = 1
    _LR_LOADFROMFILE = 0x00000010
    _IMAGE_ICON = 1

    def _build_multires_ico(png_path, ico_path):
        """pygame.display.set_icon() converts a single Surface into ONE
        Windows HICON (via SDL's CreateIconIndirect), and Windows itself
        has to scale that one bitmap down for every place it actually
        shows an icon (title bar, taskbar, Alt-Tab, each a different
        size) - its built-in icon scaler is low quality, which is
        exactly why a sharp 512x512 source still looks blurry/
        pixelated once it's on screen. A real .ico file instead embeds
        several PRE-RESIZED images at the standard sizes Windows asks
        for, so the OS just picks the matching one directly with no
        runtime scaling needed at all - this builds that .ico (cached
        to disk, regenerated only if missing or older than the source
        PNG) with each size resized individually using high-quality
        LANCZOS resampling, rather than trusting Pillow's ICO writer to
        resize adequately on its own."""
        if os.path.exists(ico_path) and os.path.getmtime(ico_path) >= os.path.getmtime(png_path):
            return
        from PIL import Image
        source = Image.open(png_path).convert("RGBA")
        sizes = _ICO_SIZES
        frames_by_size = {
            size: (source if size == source.width else source.resize((size, size), Image.LANCZOS))
            for size in sizes
        }
        # Pillow's ICO writer (IcoImagePlugin._save) uses whichever
        # image is passed as the base `im` argument as an upper bound -
        # any requested size bigger than the base gets silently
        # skipped, not resized up. The base has to be the LARGEST
        # frame (everything else in `sizes` is then <= it) for all of
        # them to actually make it into the file - passing the
        # smallest frame as the base here previously meant every size
        # past it was silently dropped, producing a "multi-size" .ico
        # that in practice only had one 16x16 frame in it.
        largest = max(sizes)
        base = frames_by_size[largest]
        other_frames = [frames_by_size[size] for size in sizes if size != largest]
        base.save(ico_path, format="ICO", sizes=[(size, size) for size in sizes], append_images=other_frames)

    def _set_crisp_window_icon(hwnd, ico_path, largest_size):
        """Applies ico_path as BOTH the small (title bar) and big
        (Alt-Tab, and - see below - taskbar) window icons via
        WM_SETICON directly.

        The small icon requests the EXACT pixel size Windows itself
        reports wanting for it (GetSystemMetrics(SM_CXSMICON), title
        bars really are that modest), so it gets the matching pre-
        resized frame out of the .ico with zero scaling. The BIG icon
        deliberately does NOT do the equivalent GetSystemMetrics
        (SM_CXICON) lookup, even though that seems like the analogous
        "ask for the exact size" move - SM_CXICON is a legacy metric
        that reports a conservative 32x32 on modern Windows, nowhere
        near the size the Windows 10/11 taskbar actually renders its
        buttons at. Confirmed as a real, visible bug: with SM_CXICON,
        the RUNNING window's taskbar icon was noticeably blurrier than
        the exe's own PINNED taskbar icon, because a pin reads the icon
        straight out of the exe's embedded PE resource (arbitrary
        resolution) while WM_SETICON's big icon was a 32x32 source the
        taskbar then had to stretch up. Requesting the LARGEST frame in
        the .ico here instead means Windows always has the
        highest-quality source to downscale FROM for whatever a given
        UI surface (taskbar at whatever DPI, Alt-Tab, jump lists) turns
        out to actually need - downscaling a big source always looks
        at least as good as an exact match, unlike upscaling a small
        one.

        Falls back silently (leaving whatever pygame.display.set_icon()
        already set) if anything here fails; this is a visual nicety,
        not something worth crashing startup over."""
        user32 = ctypes.windll.user32
        try:
            cx_small = user32.GetSystemMetrics(_SM_CXSMICON)
            cy_small = user32.GetSystemMetrics(_SM_CYSMICON)

            hicon_small = user32.LoadImageW(
                None, ico_path, _IMAGE_ICON, cx_small, cy_small, _LR_LOADFROMFILE
            )
            hicon_big = user32.LoadImageW(
                None, ico_path, _IMAGE_ICON, largest_size, largest_size, _LR_LOADFROMFILE
            )

            if hicon_small:
                user32.SendMessageW(hwnd, _WM_SETICON, _ICON_SMALL, hicon_small)
            if hicon_big:
                user32.SendMessageW(hwnd, _WM_SETICON, _ICON_BIG, hicon_big)
        except OSError as e:
            print(f"Warning: couldn't apply crisp window icon: {e}")


class WindowManager:
    def __init__(self, width=800, height=600, title="RatWar", fullscreen=True):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        icon_path = resource_path("Assets/icon/icon.png")
        icon_image = pygame.image.load(icon_path)
        pygame.display.set_icon(icon_image)

        # Remembered so F11 (toggle_fullscreen) and a future windowed
        # switch have a sane size to fall back to, rather than whatever
        # the desktop resolution happened to be - toggle_fullscreen()
        # only flips the FULLSCREEN flag on the existing surface, it
        # doesn't resize, so this is the size that sticks once someone
        # toggles back out of fullscreen.
        self.windowed_width = width
        self.windowed_height = height
        self.title = title
        self.flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE

        if fullscreen:
            # size (0, 0) with FULLSCREEN tells SDL to use the current
            # desktop resolution rather than a fixed one - the right
            # default for "launch fullscreen" since it matches whatever
            # display the game happens to start on instead of assuming
            # a specific resolution.
            self.width, self.height = 0, 0
            self.flags |= pygame.FULLSCREEN
        else:
            self.width, self.height = width, height

        self.vsync = 1
        self.screen = pygame.display.set_mode(
            (self.width, self.height), self.flags, vsync=self.vsync
        )
        # set_mode with (0, 0) resolves to the actual desktop resolution -
        # read it back so self.width/height (and anything computing
        # camera aspect from them) reflect reality, not the (0, 0) request.
        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption(title)

        # pygame.display.set_icon() above is a cross-platform baseline
        # (and needs to run before set_mode() on some platforms), but
        # it only ever produces ONE Windows HICON that the OS then
        # scales down at every size it actually displays - low quality
        # scaling that's exactly why a sharp source image still looks
        # blurry once it's on screen (see _build_multires_ico's
        # docstring). On Windows, replace it with a real multi-
        # resolution .ico applied directly via WM_SETICON so the OS
        # never has to scale anything at all.
        if sys.platform == "win32":
            try:
                ico_path = resource_path("Assets/icon/icon.ico")
                _build_multires_ico(icon_path, ico_path)
                hwnd = pygame.display.get_wm_info()["window"]
                _set_crisp_window_icon(hwnd, ico_path, max(_ICO_SIZES))
            except Exception as e:
                print(f"Warning: couldn't build/apply crisp window icon, keeping the default: {e}")

        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        pygame.event.clear(pygame.MOUSEMOTION)
        pygame.mouse.get_rel()

        self.ctx = moderngl.create_context()
        self.clock = pygame.time.Clock()
        self._fps_display_timer = 0.0
        self._last_time = time.perf_counter()

    def toggle_vsync(self):
        self.vsync = 0 if self.vsync else 1
        self.screen = pygame.display.set_mode(
            (self.width, self.height), self.flags, vsync=self.vsync
        )

    def handle_events(self, camera):
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        self.clock.tick()

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
                self.screen = pygame.display.set_mode(
                    (self.width, self.height), self.flags, vsync=self.vsync
                )
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
