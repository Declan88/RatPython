"""
Keeps the main (game + render) thread on the machine's fastest logical CPUs - but only on machines
where that actually matters.

Hybrid CPUs (Intel P/E/LP-E cores) run the same single-threaded Python code at very different
speeds - measured on a Core Ultra 9 185H: 62 ms vs 94 ms vs 144 ms for identical work, i.e. ~600 /
~400 / ~200 fps depending purely on which core the OS happened to start the main thread on. Windows 10
has no hybrid-aware scheduler, so that is a coin flip every launch.

pin_main_thread_to_fast_cores() times a tiny fixed loop on every logical CPU the process may use
(once at startup, capped at ~1.5 s), then restricts ONLY the calling thread to the CPUs within a
small margin of the fastest. It's a set, not one core, so the OS can still move the thread off a
core something else is using. Other threads (audio, Steam, driver) are untouched.

It deliberately does NOTHING unless the slowest CPU is clearly slower than the fastest
(MIN_SPREAD): on a machine whose cores are all alike - most desktops - the differences measured are
noise, and on chips with a small, deliberate clock gap between core groups (AMD X3D: the cache-heavy
CCD clocks a few % lower and is the one games WANT) picking the "fastest" would be actively wrong.
Only a big gap, which is what separate efficiency cores look like, triggers it.

Windows only. Every failure path restores the original affinity. RATWAR_NO_CPU_PIN=1 skips it.
"""

import ctypes
import os
import sys
import time

_WORK_ITERATIONS = 120_000      # ~5-8 ms per trial on a fast core
_WARMUP_SECONDS = 0.03          # spin first so the CPU clocks up before anything is measured
_KEEP_WITHIN = 1.15             # keep every CPU within 15% of the fastest
MIN_SPREAD = 1.30               # slowest/fastest below this: cores are alike, change nothing
_MIN_CPUS = 2                   # never confine the thread to fewer logical CPUs than this
_TIME_BUDGET = 1.5              # seconds; abandon (and restore) if measuring takes longer


def _spin():
    x = 0
    for i in range(_WORK_ITERATIONS):
        x += i * i % 7
    return x


def apply_priority_from_env(verbose=True):
    """RATWAR_PRIORITY=above | high (also read from ~/.ratwar/priority.txt) raises the process's scheduling priority
    class, and opts the process out of Windows' power throttling (EcoQoS - what steers background work onto
    efficiency cores). A higher priority lets the game's threads, the main thread and the graphics driver's
    own, preempt whatever else is competing for the same logical CPU. 'above' = ABOVE_NORMAL, 'high' = HIGH;
    never realtime (a spinning realtime process can starve the system, including the Steam client this game
    talks to). Returns the label applied, or '' when unset."""
    text = (os.environ.get("RATWAR_PRIORITY") or "").strip().lower()
    if not text:
        try:
            with open(os.path.join(os.path.expanduser("~"), ".ratwar", "priority.txt"), encoding="utf-8") as f:
                text = f.read().strip().lower()
        except OSError:
            text = ""
    classes = {"above": 0x00008000, "high": 0x00000080}      # ABOVE_NORMAL_PRIORITY_CLASS, HIGH_PRIORITY_CLASS
    if sys.platform != "win32" or text not in classes:
        return ""
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        process = k32.GetCurrentProcess()
        ok = bool(k32.SetPriorityClass(process, classes[text]))

        class _PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]

        # ProcessPowerThrottling (4): ControlMask bit 1 = execution speed throttling, StateMask 0 = never throttle.
        throttling = _PowerThrottling(1, 1, 0)
        k32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        un_throttled = bool(k32.SetProcessInformation(process, 4, ctypes.byref(throttling), ctypes.sizeof(throttling)))
        if verbose:
            print(f"[cpu] priority '{text}': class {'set' if ok else 'FAILED'}, "
                  f"power-throttling opt-out {'set' if un_throttled else 'unavailable'}")
        return text if ok else ""
    except Exception as e:
        print(f"[cpu] priority setting failed: {e}")
        return ""


def parse_cpu_list(text):
    """'16-19,14' / '0xF0000' -> bitmask (0 if it can't be parsed)."""
    text = (text or "").strip().lower()
    try:
        if text.startswith("0x"):
            return int(text, 16)
        mask = 0
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            lo, _, hi = part.partition("-")
            for cpu in range(int(lo), int(hi or lo) + 1):
                mask |= 1 << cpu
        return mask
    except ValueError:
        return 0


def apply_process_cpu_mask_from_env(verbose=True):
    """RATWAR_CPU_MASK (e.g. "16-19", "14,16-19" or "0xF0000") confines the WHOLE process - the main thread AND
    the threads the graphics driver creates inside it - to those logical CPUs. Call this before the window/GL
    context exists so the driver's threads are created inside the mask. For experiments: the driver's worker
    thread hands work back and forth with the main thread, and which cores they share decides how slow
    every GL call is. Returns the mask applied, or 0 (and does nothing) when the variable isn't set/valid."""
    source = "RATWAR_CPU_MASK"
    text = os.environ.get("RATWAR_CPU_MASK")
    if not text:
        # The same list from a file, so it applies to every normal launch: ~/.ratwar/cpu_mask.txt
        # containing e.g. "0,1,14-19". Delete the file (or set RATWAR_CPU_MASK to override) to undo it.
        source = "~/.ratwar/cpu_mask.txt"
        try:
            with open(os.path.join(os.path.expanduser("~"), ".ratwar", "cpu_mask.txt"), encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            text = ""
    mask = parse_cpu_list(text)
    if sys.platform != "win32" or not mask:
        return 0
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            print(f"[cpu] {source}={text!r} rejected by Windows "
                  f"(error {ctypes.get_last_error()})")
            return 0
        if verbose:
            cpus = [c for c in range(64) if mask >> c & 1]
            print(f"[cpu] {source}: whole process confined to logical CPUs {cpus}")
        return mask
    except Exception as e:
        print(f"[cpu] RATWAR_CPU_MASK failed: {e}")
        return 0


def cpu_affinity_label(tuner=None):
    """Short text for the profiler header: the process's affinity mask now, whether RATWAR_CPU_MASK /
    RATWAR_AUTO_PIN were set, and the in-game tuner's state - so every logged run says what it ran with."""
    parts = []
    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        proc, system = ctypes.c_size_t(0), ctypes.c_size_t(0)
        if k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(proc), ctypes.byref(system)):
            parts.append("process cpus " + ("all" if proc.value == system.value else hex(proc.value)))
    except Exception:
        pass
    if os.environ.get("RATWAR_CPU_MASK"):
        parts.append(f"RATWAR_CPU_MASK={os.environ['RATWAR_CPU_MASK']}")
    elif os.path.exists(os.path.join(os.path.expanduser("~"), ".ratwar", "cpu_mask.txt")):
        parts.append("cpu_mask.txt")
    if os.environ.get("RATWAR_PRIORITY") or os.path.exists(os.path.join(os.path.expanduser("~"), ".ratwar", "priority.txt")):
        parts.append("priority " + (os.environ.get("RATWAR_PRIORITY") or "file"))
    if os.environ.get("RATWAR_AUTO_PIN"):
        parts.append("auto-pin on" + ("" if tuner is None else (" (tuned)" if tuner.done else " (tuning...)")))
    return "  ".join(parts)


def choose_cpus(times):
    """times: {logical_cpu: best seconds}. Returns the CPUs to keep the thread on, or None when the
    machine's cores are too alike for pinning to be worth (or safe to) do. Pure function."""
    if len(times) < 4:
        return None
    fastest, slowest = min(times.values()), max(times.values())
    if fastest <= 0 or slowest / fastest < MIN_SPREAD:
        return None
    ranked = sorted(times, key=times.get)
    chosen = [c for c in ranked if times[c] <= fastest * _KEEP_WITHIN]
    if len(chosen) < _MIN_CPUS:
        chosen = ranked[:_MIN_CPUS]
    if len(chosen) >= len(times):
        return None                 # nothing would be excluded
    return sorted(chosen)


def pin_main_thread_to_fast_cores(verbose=True):
    """Returns the list of logical CPU numbers the thread is now restricted to, or None if nothing
    was changed."""
    if sys.platform != "win32" or os.environ.get("RATWAR_NO_CPU_PIN"):
        return None
    k32 = None
    thread = None
    original = None
    done = False
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentThread.restype = ctypes.c_void_p
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.SetThreadAffinityMask.restype = ctypes.c_size_t
        k32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        thread = k32.GetCurrentThread()

        proc_mask, sys_mask = ctypes.c_size_t(0), ctypes.c_size_t(0)
        if not k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(proc_mask), ctypes.byref(sys_mask)):
            return None
        # Only the CPUs this process is allowed on (a launcher / `start /affinity` may restrict it),
        # within the one processor group the mask covers (<= 64 CPUs; larger machines just get this group).
        cpus = [c for c in range(ctypes.sizeof(ctypes.c_size_t) * 8) if proc_mask.value >> c & 1]
        if len(cpus) < 4:
            return None                 # nothing meaningful to choose between

        start = time.perf_counter()
        times = {}
        # Two passes in opposite order, best-of per CPU: whichever CPUs happen to be measured first
        # (clocks still ramping) or hit by a background burst aren't judged on that alone.
        for pass_number, order in enumerate((cpus, cpus[::-1])):
            for cpu in order:
                previous = k32.SetThreadAffinityMask(thread, 1 << cpu)
                if not previous:
                    continue
                if original is None:
                    original = previous
                if pass_number == 0:    # (the reverse pass runs on already-warm CPUs)
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < _WARMUP_SECONDS / 3:
                        _spin()
                t = time.perf_counter()
                _spin()
                times[cpu] = min(times.get(cpu, float("inf")), time.perf_counter() - t)
                if time.perf_counter() - start > _TIME_BUDGET:
                    return None         # restored in `finally`

        chosen = choose_cpus(times)
        if chosen is None:
            if verbose and times:
                print(f"[cpu] cores are alike (slowest/fastest {max(times.values()) / min(times.values()):.2f}x "
                      f"< {MIN_SPREAD}x) - leaving thread affinity alone")
            return None
        mask = 0
        for c in chosen:
            mask |= 1 << c
        if not k32.SetThreadAffinityMask(thread, mask):
            return None
        done = True
        if verbose:
            print(f"[cpu] main thread restricted to logical CPUs {chosen} "
                  f"(fastest {min(times.values()) * 1000:.1f} ms, slowest {max(times.values()) * 1000:.1f} ms, "
                  f"{len(times) - len(chosen)} slower CPUs excluded)")
        return chosen
    except Exception as e:
        print(f"[cpu] affinity selection skipped: {e}")
        return None
    finally:
        # Anything but a successful pin puts the thread back exactly where it started.
        if not done and k32 is not None and original:
            try:
                k32.SetThreadAffinityMask(thread, original)
            except Exception:
                pass


class CpuTuner:
    """Second stage, run once with the REAL game workload. pin_main_thread_to_fast_cores() can only
    time a synthetic loop, and that can't tell apart CPUs the game itself treats very differently (on
    the machine this was written on, CPUs 10-13 gave ~450 fps and 0/14/16/18 ~700 fps although a
    plain loop times them identically - something else shares those cores: the GPU driver's
    threads, interrupts, a sibling hyper-thread...).

    So for the first few seconds of gameplay this tries each candidate CPU alone for `seconds_per_cpu`
    (median frame time, which ignores one-off hitches), then restricts the thread to the CPUs within
    `keep_within` of the best. Call update() once per rendered frame; it does nothing once finished.
    Costs a few seconds of possibly-slower frames at the start of a match, no extra work after."""

    def __init__(self, candidates, seconds_per_cpu=0.4, start_delay=2.0, keep_within=1.12, skip_frames=3):
        self.done = True
        if sys.platform != "win32" or os.environ.get("RATWAR_NO_CPU_PIN") or not candidates or len(candidates) < 3:
            return
        try:
            self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._k32.GetCurrentThread.restype = ctypes.c_void_p
            self._k32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self._k32.SetThreadAffinityMask.restype = ctypes.c_size_t
            self._thread = self._k32.GetCurrentThread()
        except Exception:
            return
        self.done = False
        self._candidates = list(candidates)
        self._mask_all = 0
        for c in self._candidates:
            self._mask_all |= 1 << c
        self._seconds = seconds_per_cpu
        self._start_delay = start_delay
        self._keep_within = keep_within
        self._skip = skip_frames
        self._created = time.perf_counter()
        self._last = None
        self._index = -1                # -1: still in the start delay
        self._frame_times = []
        self._skipped = 0
        self._switched_at = 0.0
        self.results = {}               # cpu -> median frame time (seconds)

    def update(self):
        if self.done:
            return
        now = time.perf_counter()
        last, self._last = self._last, now
        if last is None:
            return
        dt = now - last
        if self._index < 0:
            if now - self._created >= self._start_delay and dt < 0.05:
                self._switch(0, now)
            return
        if self._skipped < self._skip:      # the first frames after a move include the migration
            self._skipped += 1
            return
        if dt < 0.25:                       # a load or window drag isn't the CPU's fault
            self._frame_times.append(dt)
        if now - self._switched_at < self._seconds:
            return
        if len(self._frame_times) >= 20:
            ordered = sorted(self._frame_times)
            self.results[self._candidates[self._index]] = ordered[len(ordered) // 2]
        if self._index + 1 < len(self._candidates):
            self._switch(self._index + 1, now)
        else:
            self._finish()

    def _switch(self, index, now):
        self._index = index
        self._frame_times = []
        self._skipped = 0
        self._switched_at = now
        self._k32.SetThreadAffinityMask(self._thread, 1 << self._candidates[index])

    def _finish(self):
        self.done = True
        try:
            if len(self.results) < 3:
                self._k32.SetThreadAffinityMask(self._thread, self._mask_all)
                return
            best = min(self.results.values())
            chosen = sorted(c for c, v in self.results.items() if v <= best * self._keep_within)
            if len(chosen) < 2:
                chosen = sorted(self.results, key=self.results.get)[:2]
            mask = 0
            for c in chosen:
                mask |= 1 << c
            self._k32.SetThreadAffinityMask(self._thread, mask)
            table = ", ".join(f"{c}:{1.0 / v:.0f}" for c, v in sorted(self.results.items()))
            print(f"[cpu] in-game tuning (cpu:fps) {table} -> main thread kept on {chosen}")
        except Exception as e:
            print(f"[cpu] in-game tuning failed: {e}")
            try:
                self._k32.SetThreadAffinityMask(self._thread, self._mask_all)
            except Exception:
                pass
