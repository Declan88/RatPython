"""
Forces NVIDIA's OpenGL "Threaded optimization" on for this game's executable, through the driver's own profile
store (NVAPI DRS - what the NVIDIA Control Panel's per-program settings write).

With threaded optimization the driver runs the game's OpenGL work on a worker thread of its own. Its default,
"Auto", re-decides that by heuristics while the game runs, and on the machine this was measured on it switched
off by itself in most sessions: all GL work fell back onto the game's one Python thread, ~400 fps instead of
~850 on Boardwalk - the "random" fps halving between launches (PresentMon showed identical presentation both ways;
only the worker thread's activity differed). Forcing it on removes the coin flip.

Only the profile matching this process's own exe is touched (RatWar.exe in a build, python.exe from source), a
value someone already set there explicitly is left alone, and any failure (no NVIDIA GPU, an old driver, no
permission to save) just leaves things as they were. Call before the GL context is created.
"""

import ctypes
import os
import sys

OGL_THREAD_CONTROL_ID = 0x20C1221E
OGL_THREAD_CONTROL_ENABLE = 1
_DWORD_TYPE = 0
_OK = 0
_UNICODE_MAX = 2048
_BINARY_MAX = 4096

# nvapi_QueryInterface ids
_IDS = {
    "Initialize": 0x0150E828,
    "CreateSession": 0x0694D52E,
    "DestroySession": 0xDAD9CFF8,
    "LoadSettings": 0x375DBD6B,
    "SaveSettings": 0xFCBC7E14,
    "FindApplicationByName": 0xEEE566B2,
    "CreateProfile": 0xCC176068,
    "CreateApplication": 0x4347A9DE,
    "GetSetting": 0x73BF8338,
    "SetSetting": 0x577DD202,
}

_Unicode = ctypes.c_uint16 * _UNICODE_MAX


class _Profile(ctypes.Structure):           # NVDRS_PROFILE_V1
    _fields_ = [("version", ctypes.c_uint32), ("profileName", _Unicode), ("gpuSupport", ctypes.c_uint32),
                ("isPredefined", ctypes.c_uint32), ("numOfApps", ctypes.c_uint32),
                ("numOfSettings", ctypes.c_uint32)]


class _Application(ctypes.Structure):       # NVDRS_APPLICATION_V1
    _fields_ = [("version", ctypes.c_uint32), ("isPredefined", ctypes.c_uint32), ("appName", _Unicode),
                ("userFriendlyName", _Unicode), ("launcher", _Unicode)]


class _Binary(ctypes.Structure):
    _fields_ = [("valueLength", ctypes.c_uint32), ("valueData", ctypes.c_uint8 * _BINARY_MAX)]


class _Value(ctypes.Union):
    _fields_ = [("u32", ctypes.c_uint32), ("binary", _Binary), ("wsz", _Unicode)]


class _Setting(ctypes.Structure):           # NVDRS_SETTING_V1
    _fields_ = [("version", ctypes.c_uint32), ("settingName", _Unicode), ("settingId", ctypes.c_uint32),
                ("settingType", ctypes.c_int), ("settingLocation", ctypes.c_int),
                ("isCurrentPredefined", ctypes.c_uint32), ("isPredefinedValid", ctypes.c_uint32),
                ("predefined", _Value), ("current", _Value)]


def _version(struct):
    return ctypes.sizeof(struct) | (1 << 16)


def _unicode(text):
    value = _Unicode()
    for i, ch in enumerate(text[:_UNICODE_MAX - 1]):
        value[i] = ord(ch)
    return value


def ensure_threaded_optimization(verbose=True):
    """Sets "Threaded optimization: On" for this exe's NVIDIA profile if nothing explicit is set yet. Returns a
    short description of what happened ("" when there's nothing to do here)."""
    if sys.platform != "win32" or os.environ.get("RATWAR_NO_NV_PROFILE"):
        return ""
    try:
        nvapi = ctypes.CDLL("nvapi64.dll")
    except OSError:
        return ""                                   # no NVIDIA driver
    try:
        query = nvapi.nvapi_QueryInterface
        query.restype = ctypes.c_void_p
        query.argtypes = [ctypes.c_uint32]

        def fn(name, *argtypes):
            address = query(_IDS[name])
            if not address:
                raise RuntimeError(f"NvAPI_DRS {name} unavailable")
            return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(address)

        vp = ctypes.c_void_p
        if fn("Initialize")() != _OK:
            return ""
        session = vp()
        if fn("CreateSession", ctypes.POINTER(vp))(ctypes.byref(session)) != _OK:
            return ""
        try:
            if fn("LoadSettings", vp)(session) != _OK:
                return ""
            exe = os.path.basename(sys.executable)
            profile = vp()
            app = _Application(version=_version(_Application))
            status = fn("FindApplicationByName", vp, ctypes.POINTER(_Unicode), ctypes.POINTER(vp),
                        ctypes.POINTER(_Application))(session, ctypes.byref(_unicode(exe)), ctypes.byref(profile),
                                                      ctypes.byref(app))
            if status != _OK:
                # No profile has this exe yet: make one for it.
                info = _Profile(version=_version(_Profile), profileName=_unicode(f"Rat King ({exe})"))
                if fn("CreateProfile", vp, ctypes.POINTER(_Profile), ctypes.POINTER(vp))(
                        session, ctypes.byref(info), ctypes.byref(profile)) != _OK:
                    return ""
                new_app = _Application(version=_version(_Application), appName=_unicode(exe),
                                       userFriendlyName=_unicode("Rat King"))
                if fn("CreateApplication", vp, vp, ctypes.POINTER(_Application))(
                        session, profile, ctypes.byref(new_app)) != _OK:
                    return ""
            current = _Setting(version=_version(_Setting))
            if fn("GetSetting", vp, vp, ctypes.c_uint32, ctypes.POINTER(_Setting))(
                    session, profile, OGL_THREAD_CONTROL_ID, ctypes.byref(current)) == _OK \
                    and not current.isCurrentPredefined:
                state = "on" if current.current.u32 == OGL_THREAD_CONTROL_ENABLE else f"left at {current.current.u32}"
                return f"threaded optimization already set ({state}) for {exe}"
            setting = _Setting(version=_version(_Setting), settingId=OGL_THREAD_CONTROL_ID, settingType=_DWORD_TYPE)
            setting.current.u32 = OGL_THREAD_CONTROL_ENABLE
            if fn("SetSetting", vp, vp, ctypes.POINTER(_Setting))(session, profile, ctypes.byref(setting)) != _OK:
                return ""
            status = fn("SaveSettings", vp)(session)
            if status != _OK:
                if verbose:
                    print(f"[nvidia] couldn't save the driver profile (NvAPI status {status})")
                return ""
            if verbose:
                print(f"[nvidia] threaded optimization forced on for {exe}")
            return f"threaded optimization forced on for {exe}"
        finally:
            fn("DestroySession", vp)(session)
    except Exception as e:
        if verbose:
            print(f"[nvidia] driver profile skipped: {e}")
        return ""
