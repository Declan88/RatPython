# Patched PyInstaller bootloader (Optimus/switchable-graphics GPU hint)

## Why this exists

On a laptop with NVIDIA Optimus or AMD's equivalent switchable graphics,
Windows defaults an arbitrary `.exe` to the low-power integrated GPU. The
reliable fix for an OpenGL application (this project renders via
moderngl/pygame-ce, i.e. raw OpenGL through WGL - **not** Direct3D/DXGI) is
for the driver to find two exported symbols, `NvOptimusEnablement` and
`AmdPowerXpressRequestHighPerformance`, in the **main executable's own PE
export table**.

Two things that do **not** work, confirmed the hard way:

- **The `HKCU\Software\Microsoft\DirectX\UserGpuPreference` registry key**
  (still set by `app.py`'s `_request_high_performance_gpu()`, kept as a
  harmless extra) - this is designed for/reliably honored by DXGI apps,
  not a raw OpenGL context.
- **A companion DLL exporting the two symbols, loaded via `ctypes` at
  startup** - the symbols must be in the *main exe's* export table
  specifically; a DLL loaded into the process does nothing, no matter how
  early it's loaded.

Since PyInstaller's own bootloader binary *becomes* `app.exe` for any
frozen build, and it has no exports of its own, the only place this hint
can actually live is inside that bootloader - which means patching and
rebuilding it from PyInstaller's own C source.

## What was changed

`source_patch/main.c` is PyInstaller 6.22.2's `bootloader/src/main.c` with
two lines added near the top of the `#if defined(_WIN32)` block:

```c
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
```

`patched_windows_64bit_intel/` contains the resulting rebuilt bootloader
binaries (`run.exe`, `runw.exe`, `run_d.exe`, `runw_d.exe`) - built with
MSVC (VS2022 Build Tools) via PyInstaller's own `waf` build system
(`python ./waf all --target-arch=64bit` from the `bootloader/` directory
of the PyInstaller 6.22.2 source distribution). Verified with
`dumpbin /EXPORTS` that both symbols are present in each, and that they
survive PyInstaller's own EXE-building step (header fixing + PKG
archive appending) unchanged in the final built `app.exe`.

## How to (re)install it

**This patch lives in your local PyInstaller installation
(`site-packages`), not in this repo's Python code, and reinstalling or
upgrading PyInstaller silently restores the stock, unpatched bootloader.**
Redo this after any `pip install`/`pip install --upgrade pyinstaller`:

1. Find your PyInstaller install's bootloader folder:
   ```
   python -c "import PyInstaller, os; print(os.path.join(os.path.dirname(PyInstaller.__file__), 'bootloader', 'Windows-64bit-intel'))"
   ```
2. Copy all 4 files from `patched_windows_64bit_intel/` in this folder
   into that location, overwriting the stock ones.
3. Rebuild your exe as usual (`python -m PyInstaller ...`) - PyInstaller
   picks up whatever bootloader binaries are sitting in that folder
   automatically, no build-command flag needed.
4. Verify with `dumpbin /EXPORTS dist\app.exe` (from a Developer Command
   Prompt, or `vcvarsall.bat x64` first) - you should see
   `NvOptimusEnablement` and `AmdPowerXpressRequestHighPerformance` listed.

## If you need to reproduce this from scratch (different PyInstaller version, etc.)

1. Download the matching PyInstaller source distribution (NOT the wheel -
   the wheel only ships precompiled bootloader binaries, not the C
   source): `pip download pyinstaller==<version> --no-binary :all: --no-deps -d some_dir`, then extract the `.tar.gz`.
2. Apply the same two-line change to `bootloader/src/main.c` (see
   `source_patch/main.c` above for exactly where/what).
3. From that extracted source's `bootloader/` directory:
   `python ./waf all --target-arch=64bit` (needs MSVC - VS2022 Build
   Tools or later; the `wscript` build handles locating/invoking it,
   no manual `vcvarsall.bat` needed).
4. The rebuilt binaries land in
   `<extracted source>/PyInstaller/bootloader/Windows-64bit-intel/`.
   Install them per the steps above.
