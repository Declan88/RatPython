"""
Background sampling of the hardware state for the F9 profiler's fps timeline: each logical CPU's
actual performance (% of nominal clock - boost/throttling) from Windows' own counters, and the
NVIDIA GPU's clock, load, temperature, power draw and throttle reasons. Two long-running child
processes (typeperf, nvidia-smi) stream one line a second and a daemon thread per process keeps the
latest values, so asking for them from the render thread is a dict read - nothing ever blocks.

Everything is optional: a missing tool just leaves that part empty. Windows only (typeperf).
"""

import atexit
import csv
import subprocess
import sys
import threading

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_GPU_FIELDS = ("clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,pstate,clocks_throttle_reasons.active,"
               "pcie.link.gen.current,pcie.link.width.current")


class HardwareSampler:
    def __init__(self):
        self.cpu_perf = {}      # logical cpu -> % of nominal frequency (>100 = turbo)
        self.gpu = None         # dict, see _read_gpu
        self._procs = []
        self._running = False

    def start(self):
        if self._running or sys.platform != "win32":
            return
        self._running = True
        self._spawn(["typeperf", r"\Processor Information(*)\% Processor Performance", "-si", "1"], self._read_cpu)
        self._spawn(["nvidia-smi", f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader,nounits", "-l", "1"],
                    self._read_gpu)
        atexit.register(self.stop)

    def stop(self):
        self._running = False
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass
        self._procs = []

    def _spawn(self, args, reader):
        try:
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                 text=True, creationflags=_NO_WINDOW)
        except Exception:
            return
        self._procs.append(p)
        threading.Thread(target=reader, args=(p,), daemon=True).start()

    def _read_cpu(self, proc):
        columns = None
        for line in proc.stdout:
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            if columns is None:
                if row and row[0].startswith("(PDH"):
                    columns = []
                    for name in row[1:]:
                        # "\HOST\Processor Information(0,12)\% Processor Performance" -> 12
                        inst = name.split("(", 1)[-1].split(")", 1)[0]
                        part = inst.split(",")[-1]
                        columns.append(int(part) if part.isdigit() else None)
                continue
            fresh = {}
            for cpu, value in zip(columns, row[1:]):
                if cpu is not None:
                    try:
                        fresh[cpu] = float(value)
                    except ValueError:
                        pass
            if fresh:
                self.cpu_perf = fresh

    def _read_gpu(self, proc):
        for line in proc.stdout:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 9:
                continue
            try:
                temp = int(float(parts[3]))
                if temp == 0 and self.gpu:      # nvidia-smi now and then reports 0 for one sample
                    temp = self.gpu["temp"]
                self.gpu = {
                    "sm_mhz": int(float(parts[0])), "mem_mhz": int(float(parts[1])),
                    "watts": float(parts[2]) if parts[2] not in ("[N/A]", "") else 0.0,
                    "temp": temp, "util": int(float(parts[4])),
                    "pstate": parts[5], "reasons": parts[6],
                    "pcie": f"g{parts[7]}x{parts[8]}",
                }
            except ValueError:
                continue

    def summary(self, cpu):
        """One short text fragment for the timeline, e.g. 'perf 112% | gpu 2475MHz 71% 63C 38W P0 r=0x1'."""
        bits = []
        if cpu in self.cpu_perf:
            bits.append(f"perf {self.cpu_perf[cpu]:.0f}%")
        g = self.gpu
        if g:
            bits.append(f"gpu {g['sm_mhz']}MHz {g['util']}% {g['temp']}C {g['watts']:.0f}W {g['pstate']} pcie {g['pcie']} r={g['reasons']}")
        return " | ".join(bits)
