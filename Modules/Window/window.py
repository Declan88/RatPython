import os
import sys
import time
import pygame
import moderngl


# The frame rate limit runs this far above the chosen display rate (see frame_rate_limit): a limiter set exactly to
# a monitor's refresh rate lands just under it (each frame's wait rounds up a little), so the headroom keeps every
# refresh supplied with a fresh frame.
FPS_HEADROOM = 60


def monitor_refresh_rate():
    """The refresh rate (Hz) of the display the window is on - 60 if SDL can't tell."""
    try:
        rate = pygame.display.get_current_refresh_rate()
    except (pygame.error, AttributeError):
        try:
            rate = (pygame.display.get_desktop_refresh_rates() or [0])[0]
        except (pygame.error, AttributeError):
            rate = 0
    return rate if rate and rate > 0 else 60


def frame_rate_limit(display_fps, refresh_rate):
    """The frame rate to limit the game to for a chosen display rate: display_fps (0 = the monitor's refresh_rate,
    negative = uncapped -> None) plus FPS_HEADROOM."""
    if display_fps < 0:
        return None
    return (display_fps or refresh_rate) + FPS_HEADROOM


def _wait_until(deadline):
    """Returns at `deadline` (time.perf_counter): sleeps most of the way - sleep alone can overshoot by a
    millisecond or so, too coarse at hundreds of frames a second - and spins the last moment."""
    remaining = deadline - time.perf_counter()
    if remaining > 0.0015:
        time.sleep(remaining - 0.001)
    while time.perf_counter() < deadline:
        pass


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
    def __init__(self, width=800, height=600, title="Rat King", fullscreen=True):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        # No MSAA on the window itself: scenes render anti-aliased into their own offscreen target (Modules/Graphics/
        # scene_target.py) and present() resolves it into the window once. A multisampled window made that a
        # full-screen MSAA-to-MSAA copy plus a second resolve when the frame was shown. (It also avoids the
        # set_mode failure some drivers had with MSAA requested.)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        # RGBA8 like the target: a multisample resolve needs identical colour formats (the default here was RGB8 -
        # NVIDIA lets that pass, the spec and stricter drivers don't).
        pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)

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

        # Off by default (F10 toggles it, see handle_events): on the
        # Intel Arc iGPU this was developed on, wglSwapBuffers with a
        # swap interval of 1 intermittently blocked for 50-165ms even
        # though a whole frame's CPU+GPU work is ~2-3ms - measured as
        # ~6% of frames over 33ms with vsync on (in fullscreen, borderless
        # and windowed alike, and unchanged by MSAA or extra GPU load),
        # vs ~0.01% with it off. It costs screen tearing instead, but
        # this scene renders at ~500fps uncapped so tears are very brief.
        self.vsync = 0
        self.screen = pygame.display.set_mode((self.width, self.height), self.flags, vsync=self.vsync)
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
        # The display's refresh rate (re-read when the window moves to another display) and the frame rate flip()
        # holds the game to: None = uncapped. The game sets max_fps from its setting (see frame_rate_limit).
        self.refresh_rate = monitor_refresh_rate()
        self.max_fps = None
        self._next_frame = None

    def reset_frame_timer(self):
        """Call this after any deliberate blocking stretch on the main
        thread that ISN'T normal frame work - a scene's blocking first-
        time construction (see app.py's get_or_load_scene/loading_
        screen.py) is the actual case this exists for. Without it, the
        NEXT handle_events() call computes dt as time-since-the-frame-
        BEFORE-loading-started, so a multi-second scene load would
        appear as a single multi-second dt on the very next frame -
        physics already clamps a huge dt on its own (see physics_
        world.py's own _PHYSICS_MAX_STEP_DT), but nothing else reading
        dt directly (animation blending, footstep timers, ...) does,
        so left alone this would show up as those systems jumping far
        ahead in one tick instead of a smooth continuation from "now"."""
        self._last_time = time.perf_counter()
        # Clicks made while loading would otherwise all land on whatever is under the mouse now.
        pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEWHEEL])

    def toggle_vsync(self):
        self.vsync = 0 if self.vsync else 1
        self.screen = pygame.display.set_mode(
            (self.width, self.height), self.flags, vsync=self.vsync
        )

    def handle_events(self, camera, on_key_down=None, event_filter=None):
        """on_key_down: optional callback(pygame_key_constant), invoked
        for every KEYDOWN event this method sees (after its own internal
        handling below, which happens regardless - it always still
        toggles fullscreen/vsync/quits on its own reserved keys even if
        a caller is also watching for other ones). Lets a caller react
        to an edge-triggered key press (e.g. a mode toggle) without this
        generic window/input module needing to know anything about
        game-specific concepts like "third person mode" - it only ever
        hands back which key went down, callers decide what that means.
        Not called for QUIT/ESCAPE, since those already end the loop
        before reaching it.

        event_filter: optional callback(event) -> bool, offered every
        event except QUIT first; returning True marks it consumed and
        skips this method's own handling of it (the UI uses this to keep
        mouse motion from turning the camera while a menu has the cursor)."""
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        self.clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, dt
            if event_filter is not None and event_filter(event):
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, dt
                elif event.key == pygame.K_F11:
                    pygame.display.toggle_fullscreen()
                    self.refresh_rate = monitor_refresh_rate()
                elif event.key == pygame.K_F10:
                    self.toggle_vsync()
                if on_key_down is not None:
                    on_key_down(event.key)
            elif event.type == pygame.MOUSEMOTION:
                camera.process_mouse(event.rel[0], event.rel[1])
            elif event.type == pygame.WINDOWDISPLAYCHANGED:
                self.refresh_rate = monitor_refresh_rate()     # moved to another monitor
            elif event.type == pygame.VIDEORESIZE:
                # Just the new size: an OpenGL window's drawable follows the window by itself. NOT set_mode again -
                # with FULLSCREEN in self.flags that re-entered fullscreen on every resize: on macOS each re-entry
                # is a new fullscreen transition, which sends another resize, so the window flipped in and out of
                # fullscreen endlessly (and leaving fullscreen with F11 was undone by the resize it caused).
                self.width, self.height = event.w, event.h
                self.ctx.viewport = (0, 0, self.width, self.height)
                camera.aspect = self.width / self.height

        self._fps_display_timer += dt
        if self._fps_display_timer >= 0.5:
            self._fps_display_timer = 0.0
            vsync_label = ("vsync on" if self.vsync else
                           f"limit {self.max_fps:.0f}" if self.max_fps else "uncapped")
            pygame.display.set_caption(
                f"{self.title} - {self.clock.get_fps():.0f} FPS - {vsync_label}"
            )
        return True, dt

    def flip(self):
        pygame.display.flip()
        self._limit_frame_rate()

    def _limit_frame_rate(self):
        """Holds frames to max_fps (nothing with vsync on, which paces them itself, or uncapped): waits out the
        rest of this frame's slot. Slots follow each other exactly while the game keeps up; one that runs late
        starts the next slot from now instead of rushing to catch up."""
        if not self.max_fps or self.vsync:
            self._next_frame = None
            return
        period = 1.0 / self.max_fps
        now = time.perf_counter()
        if self._next_frame is not None and self._next_frame > now:
            _wait_until(self._next_frame)
            now = time.perf_counter()
        late = self._next_frame is None or now - self._next_frame > period
        self._next_frame = (now if late else self._next_frame) + period

    def quit(self):
        pygame.quit()
