"""
Runs a command (the game, by default) with its CPU limited, to see how it behaves on a slower
PC. Windows only; changes nothing outside the launched process.

    python run_limited.py                       # "modest": 4 cores, 1.0 core's worth of time in total
    python run_limited.py --preset mild         # a bit slower than this PC
    python run_limited.py --preset weak         # an old/budget PC
    python run_limited.py --cores 1 --speed 0.4 # your own numbers
    python run_limited.py --speed 0.5 -- python app.py
    python run_limited.py --eco                 # also ask Windows to run it as a low-power process

Presets (cores / speed): mild 6 / 1.3, modest 4 / 1.0, budget 3 / 0.85, weak 2 / 0.7, very-weak 2 / 0.45.
The game's main thread uses about one core plus a bit for the graphics driver, audio and Steam,
so a total ("speed") near 1.3 barely touches it; around 1.0 it has roughly the speed of a mid-range
PC's core, and below that the game itself starts to slow down.

--cores  how many logical cores the process may run on (its affinity)
--speed  the share of ONE core's time it's allowed, in total, across ALL its threads (1.0 = a full
         core, 0.5 = half; the game itself uses a bit over 1.0 counting the driver/audio/Steam threads): enforced by a Windows job object's hard CPU cap, so the game genuinely
         runs slower rather than just being scheduled later. Lower = a weaker CPU. The cap works in
         short time slices, so expect some stutter at low values - that's part of being throttled.
--share  N busy-loop processes compete with the game on its cores: a smoother, steadier slowdown
         than --speed (e.g. --cores 2 --share 2 leaves the game about half of 2 cores).
--eco    marks the process for Windows' "efficiency mode" (EcoQoS), which on some CPUs also lowers the
         clock speed it runs at.

The limits end with the process. Close the game (or Ctrl+C here) to stop.
"""

import argparse
import ctypes
import os
import subprocess
import sys
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

JobObjectCpuRateControlInformation = 15
JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x1
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x4
ProcessPowerThrottling = 4
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1


PRESETS = {"mild": (6, 1.3), "modest": (4, 1.0), "budget": (3, 0.85), "weak": (2, 0.7), "very-weak": (2, 0.45)}


class CpuRateControl(ctypes.Structure):
    _fields_ = [("ControlFlags", wintypes.DWORD), ("CpuRate", wintypes.DWORD)]


class PowerThrottling(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


def _check(ok, what):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), what)


def limit(process_handle, cores, speed, eco):
    """Applies the limits to an already-started process."""
    total = os.cpu_count() or 1
    if cores:
        mask = (1 << min(cores, total)) - 1
        _check(kernel32.SetProcessAffinityMask(process_handle, mask), "SetProcessAffinityMask")
    if speed:
        # CpuRate is in 1/100ths of a percent of the WHOLE machine's cpu time.
        rate = max(1, min(10000, int(speed / total * 10000)))
        job = kernel32.CreateJobObjectW(None, None)
        _check(job, "CreateJobObject")
        info = CpuRateControl(JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP, rate)
        _check(kernel32.SetInformationJobObject(job, JobObjectCpuRateControlInformation,
                                                ctypes.byref(info), ctypes.sizeof(info)),
               "SetInformationJobObject (cpu rate)")
        _check(kernel32.AssignProcessToJobObject(job, process_handle), "AssignProcessToJobObject")
        limit.job = job          # keep the handle alive as long as this launcher runs
    if eco:
        state = PowerThrottling(1, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
        _check(kernel32.SetProcessInformation(process_handle, ProcessPowerThrottling,
                                              ctypes.byref(state), ctypes.sizeof(state)),
               "SetProcessInformation (power throttling)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="modest")
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--eco", action="store_true")
    parser.add_argument("--share", type=int, default=0, metavar="N",
                        help="smoother slowdown: pin the game to --cores cores and run N busy-loop processes on the same "
                             "cores, so the game gets roughly cores/(cores+N) of them (no --speed cap is applied)")
    parser.add_argument("command", nargs="*")
    args = parser.parse_args()
    preset_cores, preset_speed = PRESETS[args.preset]
    args.cores = args.cores if args.cores is not None else preset_cores
    args.speed = args.speed if args.speed is not None else preset_speed
    command = args.command or [sys.executable, "app.py"]
    if command and command[0] == "--":
        command = command[1:]

    process = subprocess.Popen(command, cwd=os.path.dirname(os.path.abspath(__file__)))
    handle = kernel32.OpenProcess(0x1F0FFF, False, process.pid)       # PROCESS_ALL_ACCESS
    _check(handle, "OpenProcess")
    burners = []
    if args.share:
        limit(handle, args.cores, None, args.eco)
        mask = (1 << min(args.cores, os.cpu_count() or 1)) - 1
        for _ in range(args.share):
            burner = subprocess.Popen([sys.executable, "-c", "while True: pass"])
            burner_handle = kernel32.OpenProcess(0x1F0FFF, False, burner.pid)
            kernel32.SetProcessAffinityMask(burner_handle, mask)
            burners.append(burner)
        args.speed = 0
    else:
        limit(handle, args.cores, args.speed, args.eco)
    how = f"sharing with {args.share} busy process(es)" if args.share else f"{args.speed:.0%} of a core"
    print(f"running {' '.join(command)} (pid {process.pid}): {args.cores} core(s), {how}{', eco mode' if args.eco else ''}")
    try:
        sys.exit(process.wait())
    except KeyboardInterrupt:
        process.terminate()
    finally:
        for burner in burners:
            burner.kill()


if __name__ == "__main__":
    main()
