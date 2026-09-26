"""
A tiny in-game frame profiler: press F9 to show, per second, where each frame's time goes
(events, physics/animation, weapons, network, rendering, presenting...) plus the network's own
breakdown and packet counts. Off, every call is a single attribute check.

While on it also appends a block to ~/.ratwar/frame_profile.txt every ten seconds, headed with
the machine's role (HOST/CLIENT) and how many other players are connected - so the same
session run on two machines can be compared side by side.
"""

import os
import time

_LOG_INTERVAL = 10.0


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

    def set_enabled(self, on):
        self.enabled = bool(on)
        self._acc.clear()
        self._order.clear()
        self._frames = 0
        self._counter_base = dict(self.counters)
        self._window_start = self._last_log = time.perf_counter()
        self.lines = ["profiling..."] if on else []

    def begin(self):
        if self.enabled:
            self._t = time.perf_counter()

    def mark(self, name):
        """Charges the time since the previous mark (or begin) to `name`."""
        if not self.enabled:
            return
        now = time.perf_counter()
        self.add(name, now - self._t)
        self._t = now

    def add(self, name, seconds):
        """Charges seconds to `name` without touching the running clock - for timings taken
        inside a section that mark() also covers (shown indented, as a breakdown)."""
        if not self.enabled:
            return
        if name not in self._acc:
            self._acc[name] = 0.0
            self._order.append(name)
        self._acc[name] += seconds

    def count(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def end_frame(self):
        if not self.enabled:
            return
        self._frames += 1
        now = time.perf_counter()
        span = now - self._window_start
        if span < 1.0:
            return
        frames = max(self._frames, 1)
        total = sum(v for k, v in self._acc.items() if not k.startswith("  "))
        lines = [f"{frames / span:6.0f} fps   {span / frames * 1000:6.2f} ms/frame"]
        for name in self._order:
            lines.append(f"{name:<26s} {self._acc[name] / frames * 1000:6.3f} ms")
        rates = [f"{k} {(v - self._counter_base.get(k, 0)) / span:.0f}/s" for k, v in self.counters.items()]
        if rates:
            lines.append("   ".join(rates))
        self.lines = lines
        self._counter_base = dict(self.counters)
        self._acc = {k: 0.0 for k in self._acc}
        self._frames = 0
        self._window_start = now
        if now - self._last_log >= _LOG_INTERVAL:
            self._last_log = now
            self._append_log(lines)

    def _append_log(self, lines):
        try:
            folder = os.path.join(os.path.expanduser("~"), ".ratwar")
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "frame_profile.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}  {self.header()}\n")
                f.write("\n".join(lines) + "\n")
        except OSError:
            pass
