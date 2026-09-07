import ctypes
import os

dll_path = r"E:\pull2\steam_api64.dll"
ctypes.CDLL(dll_path, mode=ctypes.RTLD_GLOBAL)

import py_steam_net
print([m for m in dir(py_steam_net.PySteamClient) if not m.startswith("_")])