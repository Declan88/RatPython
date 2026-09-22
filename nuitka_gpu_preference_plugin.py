"""Nuitka user plugin (see BuildCMD_Nuitka's own --user-plugin= flag) -
the actual fix for the Optimus/switchable-graphics "runs on the
integrated GPU instead of the dedicated one" issue app.py's own
_request_high_performance_gpu()/its surrounding NOTE comment already
diagnosed but couldn't finish: the registry hint only works for D3D
apps, and NVIDIA/AMD's driver-level Optimus/PowerXpress check requires
the two well-known globals below (NvOptimusEnablement,
AmdPowerXpressRequestHighPerformance) to be real PE-EXPORTED symbols in
the ACTUAL RUNNING .exe's own export table - not a companion DLL loaded
at runtime (confirmed, per that same comment, to do nothing).

Confirmed working via an isolated test build (a throwaway hello-world
script, --onefile, this exact plugin) and pefile inspection of the
result:

  >>> import pefile
  >>> pe = pefile.PE("hello.exe")
  >>> [s.name for s in pe.DIRECTORY_ENTRY_EXPORT.symbols]
  [b'AmdPowerXpressRequestHighPerformance', b'NvOptimusEnablement']

Two things had to be gotten right, neither of which is obvious from
Nuitka's own docs:

1. getExtraCodeFiles()'s key MUST contain "onefile_" for a --onefile
   build specifically. Nuitka's own Plugins.py filters extra code files
   by whether "onefile_" appears in the key, keeping each file OUT of
   whichever compile stage doesn't match (see Plugins.py's own
   _getExtraCodeFiles: `if (for_onefile and "onefile_" not in key) or
   (not for_onefile and "onefile_" in key): continue`). A --onefile
   build compiles TWO separate binaries: the actual program logic ends
   up in a DLL (confirmed: the standalone dist folder contains a
   `<name>.dll`, not a `<name>.exe`), while the ONLY real .exe is the
   onefile bootstrap/launcher. A key without "onefile_" lands the code
   in that DLL - exactly the "DLL loaded at runtime does nothing" case
   already ruled out. Only an "onefile_"-prefixed key reaches the
   bootstrap .exe's own compilation, where the symbols actually need to
   be.

2. The onefile bootstrap, once launched, re-executes ITSELF as a child
   process to actually run the unpacked program (confirmed via Get-
   Process during a live test run: both the parent and child process's
   own .exe Path were the identical bootstrap .exe file, e.g.
   "hello2.exe", not two different files - sys.executable inside the
   child reports a virtual path into the temp extraction directory, but
   the actual loaded PE image Windows/the GPU driver sees is still that
   same bootstrap .exe). So exporting these symbols from the bootstrap
   .exe covers the process that actually creates the OpenGL context too,
   not just the short-lived parent - there's no separate "real" exe file
   anywhere else in a --onefile build that would need this instead.
"""

from nuitka.plugins.PluginBase import NuitkaPluginBase


class NuitkaPluginGpuPreference(NuitkaPluginBase):
    plugin_name = "gpu-preference"
    plugin_desc = (
        "Export NvOptimusEnablement/AmdPowerXpressRequestHighPerformance from the "
        "built exe, so Optimus/switchable-graphics laptops launch it on the "
        "dedicated GPU instead of the integrated one."
    )

    @classmethod
    def isRelevant(cls):
        # Only meaningful on Windows (these two symbols are a Windows-
        # driver-specific mechanism - see this module's own docstring) -
        # a no-op plugin on any other platform, rather than failing the
        # build or emitting a pointless extra code file there.
        import sys
        return sys.platform == "win32"

    def getExtraCodeFiles(self):
        # "onefile_" in the key is required - see this module's own
        # docstring, point 1, for exactly why a key without it silently
        # ends up in the wrong binary (the standalone DLL, not the
        # bootstrap .exe).
        return {
            "onefile_gpu_preference.c": (
                "// NVIDIA Optimus - see NVIDIA's own published guidance\n"
                "// on this exact mechanism (\"Enabling High Performance "
                "Graphics Rendering on Optimus Systems\").\n"
                "__declspec(dllexport) int NvOptimusEnablement = 1;\n"
                "// AMD PowerXpress - the vendor-neutral equivalent for "
                "AMD switchable-graphics laptops.\n"
                "__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;\n"
            )
        }
