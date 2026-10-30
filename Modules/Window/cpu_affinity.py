"""
Keeps the game on the right CPU cores of a hybrid (Intel P/E-core) CPU.

On such CPUs where a thread runs changes everything, and Windows 10 doesn't know which cores are fast (Windows 11
uses Intel's Thread Director; Windows 10 never got it). Measured on a Core Ultra 9 185H, Boardwalk:
  * the main thread: ~920 fps on any P-core, ~610 on an E-core, ~405 on a low-power E-core;
  * the graphics driver's own worker thread (~85% of a core, created inside this process) matters as much -
    with the main thread on a P-core: ~925 fps with the driver on another P-core, ~710 on the main thread's
    hyper-thread sibling, ~600 on E-cores.
Earlier pinning only moved the main thread, so a launch whose driver thread landed on an E-core stayed slow.

prefer_performance_cores() therefore restricts the whole process to the P-cores (their type read from Windows, nothing
timed) and gives the main thread half of them; keep_other_threads_off_main_cores() then moves every other thread -
the driver's worker above all - onto the other half, so the two never share a physical core. Hard affinity: Windows
CPU sets (the "soft" API) were tested and simply ignored. Does nothing on a CPU whose cores are all alike.
RATWAR_NO_CPU_PIN=1 skips it.

RATWAR_CPU_MASK / RATWAR_PRIORITY remain for experiments (see their functions).
"""

import ctypes
import os
import sys
from ctypes import wintypes

_applied = ""    # what prefer_performance_cores did, for the profiler header
_layout = None   # (main thread id, main thread's mask, everything else's mask) once applied


class _CpuSetInformation(ctypes.Structure):
    """SYSTEM_CPU_SET_INFORMATION (the CpuSet member of its union)."""
    _fields_ = [
        ("Size", wintypes.DWORD), ("Type", wintypes.DWORD), ("Id", wintypes.DWORD), ("Group", wintypes.WORD),
        ("LogicalProcessorIndex", ctypes.c_ubyte), ("CoreIndex", ctypes.c_ubyte),
        ("LastLevelCacheIndex", ctypes.c_ubyte), ("NumaNodeIndex", ctypes.c_ubyte),
        ("EfficiencyClass", ctypes.c_ubyte), ("AllFlags", ctypes.c_ubyte), ("SchedulingClass", ctypes.c_ubyte),
        ("_reserved", ctypes.c_ubyte), ("AllocationTag", ctypes.c_ulonglong),
    ]


def _cpu_sets(k32):
    """Every logical CPU's (cpu-set id, logical index, physical core, efficiency class)."""
    needed = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
    buffer = ctypes.create_string_buffer(needed.value)
    if not k32.GetSystemCpuSetInformation(buffer, needed, ctypes.byref(needed), None, 0):
        return []
    sets, offset = [], 0
    size = ctypes.sizeof(_CpuSetInformation)
    while offset < needed.value:
        info = _CpuSetInformation.from_buffer_copy(buffer.raw[offset:offset + size])
        if info.Type == 0:    # CpuSetInformation
            sets.append((info.Id, info.LogicalProcessorIndex, info.CoreIndex, info.EfficiencyClass))
        offset += info.Size or size
    return sets


def _kernel32():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GetCurrentThread.restype = wintypes.HANDLE
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetThreadAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetThreadAffinityMask.restype = ctypes.c_size_t
    k32.OpenThread.restype = wintypes.HANDLE
    k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32


def _mask(cpus):
    mask = 0
    for cpu in cpus:
        mask |= 1 << cpu
    return mask


def prefer_performance_cores(verbose=True):
    """On a hybrid CPU: confines the process to the P-cores and the calling (main) thread to half of them - call it
    from the main thread before the window/GL context exists, then keep_other_threads_off_main_cores() once the
    driver's threads exist. Returns a description, or "" if nothing was changed."""
    global _applied, _layout
    if sys.platform != "win32" or os.environ.get("RATWAR_NO_CPU_PIN"):
        return ""
    try:
        k32 = _kernel32()
        sets = [s for s in _cpu_sets(k32) if s[1] < ctypes.sizeof(ctypes.c_size_t) * 8]   # processor group 0
        classes = {cls for _, _, _, cls in sets}
        if len(classes) < 2:
            return ""    # every core alike: Windows' own placement is fine
        fastest = max(classes)
        cores = sorted({core for _, _, core, cls in sets if cls == fastest})
        if len(cores) >= 2:
            # The main thread on the upper half (away from core 0, where most hardware interrupts land).
            main_cores, rest_cores = set(cores[len(cores) // 2:]), set(cores[:len(cores) // 2])
        else:
            main_cores = rest_cores = set(cores)
        main_cpus = sorted(cpu for _, cpu, core, cls in sets if cls == fastest and core in main_cores)
        rest_cpus = sorted(cpu for _, cpu, core, cls in sets if cls == fastest and core in rest_cores)
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), _mask(main_cpus) | _mask(rest_cpus)):
            print(f"[cpu] process affinity rejected ({ctypes.get_last_error()}) - leaving placement alone")
            return ""
        if not k32.SetThreadAffinityMask(k32.GetCurrentThread(), _mask(main_cpus)):
            print(f"[cpu] main thread affinity rejected ({ctypes.get_last_error()})")
        _layout = (k32.GetCurrentThreadId(), _mask(main_cpus), _mask(rest_cpus))
        _applied = f"main thread on P-cpus {main_cpus}, rest on {rest_cpus}"
        if verbose:
            print(f"[cpu] hybrid CPU: {_applied}")
        return _applied
    except Exception as e:
        print(f"[cpu] CPU placement skipped: {e}")
        return ""


class _ThreadEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", ctypes.c_long),
                ("tpDeltaPri", ctypes.c_long), ("dwFlags", wintypes.DWORD)]


def keep_other_threads_off_main_cores():
    """Moves every thread of this process except the main one onto the cores prefer_performance_cores left for
    them. The graphics driver starts its threads lazily, so call this after the GL context exists and again after
    a load. A no-op if prefer_performance_cores didn't run. Returns how many threads were moved."""
    if _layout is None:
        return 0
    main_id, _, rest_mask = _layout
    moved = 0
    try:
        k32 = _kernel32()
        snapshot = k32.CreateToolhelp32Snapshot(0x4, 0)     # TH32CS_SNAPTHREAD
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        ok = k32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == os.getpid() and entry.th32ThreadID != main_id:
                handle = k32.OpenThread(0x0020 | 0x0040, False, entry.th32ThreadID)   # SET_ | QUERY_INFORMATION
                if handle:
                    moved += bool(k32.SetThreadAffinityMask(handle, rest_mask))
                    k32.CloseHandle(handle)
            ok = k32.Thread32Next(snapshot, ctypes.byref(entry))
        k32.CloseHandle(snapshot)
    except Exception as e:
        print(f"[cpu] moving the other threads failed: {e}")
    return moved


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
    """RATWAR_CPU_MASK (e.g. "16-19", "14,16-19" or "0xF0000"; or ~/.ratwar/cpu_mask.txt) confines the WHOLE
    process - the main thread and the driver's threads - to those logical CPUs, for experiments; it replaces
    prefer_performance_cores. Call before the window/GL context exists. Returns the mask applied, or 0."""
    source = "RATWAR_CPU_MASK"
    text = os.environ.get("RATWAR_CPU_MASK")
    if not text:
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


def cpu_affinity_label():
    """Short text for the profiler header: how this run's CPUs were set up, so every logged run says it."""
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
    if _applied:
        parts.append(_applied)
    if os.environ.get("RATWAR_CPU_MASK"):
        parts.append(f"RATWAR_CPU_MASK={os.environ['RATWAR_CPU_MASK']}")
    elif os.path.exists(os.path.join(os.path.expanduser("~"), ".ratwar", "cpu_mask.txt")):
        parts.append("cpu_mask.txt")
    if os.environ.get("RATWAR_PRIORITY") or os.path.exists(os.path.join(os.path.expanduser("~"), ".ratwar", "priority.txt")):
        parts.append("priority " + (os.environ.get("RATWAR_PRIORITY") or "file"))
    return "  ".join(parts)
