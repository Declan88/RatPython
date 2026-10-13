"""
A tiny in-game frame profiler: press F9 to show, per second, where each frame's time goes
(events, physics/animation, weapons, network, rendering, presenting...) plus the network's own
breakdown and packet counts. Off, every call is a single attribute check.

While on it also appends a block to ~/.ratwar/frame_profile.txt every ten seconds, headed with
the machine's role (HOST/CLIENT) and how many other players are connected - so the same
session run on two machines can be compared side by side.
"""

import atexit
import ctypes
import gc
import os
import sys
import time
from collections import Counter

from .hardware_sampler import HardwareSampler

_LOG_INTERVAL = 10.0
# A frame whose wall-clock time (end of the last frame to end of this one) exceeds this is a hitch:
# logged individually, with the time no stage mark covers (GC, OS, driver) and any GC that ran.
# Anything past _HITCH_MAX_MS is a load / window drag, not a stutter, and is skipped.
_HITCH_MS = 25.0
_HITCH_MAX_MS = 1000.0


class FrameProfiler:
    def __init__(self):
        self.enabled = False
        self._t = 0.0
        self._acc = {}
        self._order = []
        self._frames = 0
        self._window_start = 0.0
        self._last_log = 0.0
        self.lines = []          # the last finished window, one string per line
        self.header = lambda: ""  # callable -> extra header text (role, players); set by the game
        self.counters = {}       # name -> running count (packets in...), rate shown per second
        self._counter_base = {}
        self._marks = {}         # this frame's stage times (mark() only)
        self._subs = {}          # this frame's add() timings (the indented sub-stages)
        self._ignore = False
        self._worst = (0.0, {})  # the slowest frame of the window: total seconds, stage times
        self._last_end = 0.0     # wall-clock end of the previous frame (hitch detection)
        self._hitches = []       # formatted hitch lines waiting for the next log flush
        self._timeline = []      # one "time fps cpu" line per second, flushed with the hitches
        self.extra = lambda: ""   # callable -> text appended to each timeline line (set by the game)
        self.hardware = HardwareSampler()
        atexit.register(self.flush)
        self._cpu_samples = Counter()   # logical CPU the main thread was on, sampled once per frame
        self._get_cpu = None
        if sys.platform == "win32":
            try:
                self._get_cpu = ctypes.WinDLL("kernel32").GetCurrentProcessorNumber
            except Exception:
                pass
        self._gc_start = 0.0
        self._gc_frame = []      # (generation, ms, collected) for every GC that ran this frame

    def _gc_callback(self, phase, info):
        if phase == "start":
            self._gc_start = time.perf_counter()
        else:
            self._gc_frame.append((info.get("generation", -1), (time.perf_counter() - self._gc_start) * 1000.0,
                                   info.get("collected", 0)))

    def flush(self):
        """Writes anything still buffered (the last seconds of a session) - registered to run at exit."""
        self._flush_hitches()

    def set_enabled(self, on):
        on = bool(on)
        if on:
            self.hardware.start()
        else:
            self.hardware.stop()
        if on and self._gc_callback not in gc.callbacks:
            gc.callbacks.append(self._gc_callback)
        elif not on:
            if self._gc_callback in gc.callbacks:
                gc.callbacks.remove(self._gc_callback)
            self._flush_hitches()
        self._last_end = 0.0
        self._gc_frame = []
        self.enabled = on
        self._acc.clear()
        self._order.clear()
        self._frames = 0
        self._worst = (0.0, {})
        self._counter_base = dict(self.counters)
        self._window_start = self._last_log = time.perf_counter()
        self.lines = ["profiling..."] if on else []

    def begin(self):
        if self.enabled:
            self._t = time.perf_counter()
            self._marks = {}
            self._subs = {}

    def mark(self, name):
        """Charges the time since the previous mark (or begin) to `name`."""
        if not self.enabled:
            return
        now = time.perf_counter()
        self.add(name, now - self._t, sub=False)
        self._marks[name] = self._marks.get(name, 0.0) + now - self._t
        self._t = now

    def add(self, name, seconds, sub=True):
        """Charges seconds to `name` without touching the running clock - for timings taken
        inside a section that mark() also covers (shown indented, as a breakdown)."""
        if not self.enabled:
            return
        if name not in self._acc:
            self._acc[name] = 0.0
            self._order.append(name)
        self._acc[name] += seconds
        if sub:
            self._subs[name] = self._subs.get(name, 0.0) + seconds

    def ignore_frame(self):
        """Keeps this frame out of the 'worst frame' line (a spike the profiler itself caused)."""
        self._ignore = True

    def count(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def end_frame(self):
        if not self.enabled:
            return
        self._frames += 1
        if self._get_cpu is not None:
            self._cpu_samples[self._get_cpu()] += 1
        now = time.perf_counter()
        frame_total = sum(self._marks.values())
        self._note_hitch(now, frame_total)
        if frame_total > self._worst[0] and not self._ignore:
            self._worst = (frame_total, self._marks)
        self._ignore = False
        span = now - self._window_start
        if span < 1.0:
            return
        frames = max(self._frames, 1)
        cpu_text = ""
        if self._cpu_samples:
            n = sum(self._cpu_samples.values())
            cpu_text = "  cpu " + " ".join(f"{c}:{100 * v // n}%" for c, v in self._cpu_samples.most_common(3))
            hw = self.hardware.summary(self._cpu_samples.most_common(1)[0][0])
            if hw:
                cpu_text += "  | " + hw
        try:
            extra = self.extra()
        except Exception:
            extra = ""
        if extra:
            cpu_text += "  | " + extra
        flip = self._acc.get("flip (GPU wait / vsync)")
        flip_text = f"  flip {flip / frames * 1000:5.3f}ms" if flip is not None else ""
        self._timeline.append(
            f"{time.strftime('%H:%M:%S')}  {frames / span:5.0f} fps  worst {self._worst[0] * 1000:5.1f} ms{flip_text}{cpu_text}")
        total = sum(v for k, v in self._acc.items() if not k.startswith("  "))
        lines = [f"{frames / span:6.0f} fps   {span / frames * 1000:6.2f} ms/frame"]
        for name in self._order:
            lines.append(f"{name:<26s} {self._acc[name] / frames * 1000:6.3f} ms")
        worst_total, worst_marks = self._worst
        if worst_marks:
            top = sorted(worst_marks.items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"worst frame {worst_total * 1000:.2f} ms: " + ", ".join(f"{k} {v * 1000:.2f}" for k, v in top))
        if self._cpu_samples:
            n = sum(self._cpu_samples.values())
            lines.append("main thread on CPU: " + ", ".join(
                f"{c} {100 * v // n}%" for c, v in self._cpu_samples.most_common(4)))
            self._cpu_samples.clear()
        rates = [f"{k} {(v - self._counter_base.get(k, 0)) / span:.0f}/s" for k, v in self.counters.items()]
        if rates:
            lines.append("   ".join(rates))
        self.lines = lines
        self._counter_base = dict(self.counters)
        self._acc = {k: 0.0 for k in self._acc}
        self._frames = 0
        self._worst = (0.0, {})
        self._window_start = now
        if now - self._last_log >= _LOG_INTERVAL:
            self._last_log = now
            self._append_log(lines)

    def _note_hitch(self, now, frame_total):
        wall = (now - self._last_end) * 1000.0 if self._last_end else 0.0
        self._last_end = now
        gcs, self._gc_frame = self._gc_frame, []
        if self._ignore or not (_HITCH_MS <= wall <= _HITCH_MAX_MS):
            return
        top = sorted(self._marks.items(), key=lambda kv: -kv[1])[:3]
        stages = ", ".join(f"{k} {v * 1000:.1f}" for k, v in top)
        other = max(wall - frame_total * 1000.0, 0.0)
        subs = sorted(self._subs.items(), key=lambda kv: -kv[1])[:2]
        sub_text = ("  sub: " + ", ".join(f"{k.strip()} {v * 1000:.1f}" for k, v in subs)) if subs and subs[0][1] >= 0.002 else ""
        gc_text = ""
        if gcs:
            # one line however many collections ran: total time, and the slowest one's generation
            slowest = max(gcs, key=lambda c: c[1])
            gc_text = (f"  GC: {len(gcs)} runs, {sum(c[1] for c in gcs):.1f} ms total, "
                       f"slowest gen{slowest[0]} {slowest[1]:.1f} ms")
        self._hitches.append(
            f"{time.strftime('%H:%M:%S')}  {wall:6.1f} ms  outside-marks {other:6.1f}  [{stages}]{sub_text}{gc_text}")

    def _flush_timeline(self):
        if not self._timeline:
            return
        lines, self._timeline = self._timeline, []
        try:
            folder = os.path.join(os.path.expanduser("~"), ".ratwar")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "fps_timeline.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {self.header()}\n")
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass

    def _flush_hitches(self):
        self._flush_timeline()
        if not self._hitches:
            return
        lines, self._hitches = self._hitches, []
        try:
            folder = os.path.join(os.path.expanduser("~"), ".ratwar")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "hitches.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {self.header()}\n")
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass

    def _append_log(self, lines):
        self._flush_hitches()
        try:
            folder = os.path.join(os.path.expanduser("~"), ".ratwar")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "frame_profile.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {self.header()}\n")
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass


class FunctionProfiler:
    """F8: records every Python function call until F8 is pressed again, then writes the most
    expensive ones to ~/.ratwar/function_profile.txt. Slow while recording (expect a much lower
    fps) - it's for finding WHICH functions cost what, not for measuring fps."""

    def __init__(self):
        self.profile = None

    @property
    def active(self):
        return self.profile is not None

    def toggle(self):
        import cProfile
        if self.profile is None:
            self.profile = cProfile.Profile()
            self.profile.enable()
            return None
        self.profile.disable()
        profile, self.profile = self.profile, None
        return self._write(profile)

    @staticmethod
    def _write(profile):
        import io
        import pstats
        try:
            folder = os.path.join(os.path.expanduser("~"), ".ratwar")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, "function_profile.txt")
            out = io.StringIO()
            stats = pstats.Stats(profile, stream=out)
            out.write("== by time spent inside each function itself ==\n")
            stats.sort_stats("tottime").print_stats(45)
            out.write("\n== by time including everything it calls ==\n")
            stats.sort_stats("cumtime").print_stats(45)
            with open(path, "w", encoding="utf-8") as f:
                f.write(out.getvalue())
            return path
        except OSError:
            return None

