"""
Runtime lightmap textures stored as GL_R11F_G11F_B10F (4 bytes/texel) instead of RGB16F (6) -
a third less VRAM for a precision loss (~0.8% relative on R/G, ~1.6% on B) nobody can see in
baked lighting. Lightmaps are non-negative HDR, which is exactly what this format holds.

moderngl can't create this internal format itself, so the texture is made with a few raw GL
calls (all OpenGL 1.1 entry points, exported straight from the system GL library - no PyOpenGL)
and handed to moderngl via Context.external_texture. The driver converts the half-float data we
upload, so there's no CPU-side bit packing. moderngl's release() does NOT free an external
texture's GL object, hence the wrapper class below, which does.

create_packed_lightmap returns None on any failure (no GL library, GL error, ...) so callers
fall back to an ordinary RGB16F moderngl texture.
"""

import ctypes
import sys

import moderngl

_GL_TEXTURE_2D = 0x0DE1
_GL_UNPACK_ALIGNMENT = 0x0CF5
_GL_R11F_G11F_B10F = 0x8C3A
_GL_RGB = 0x1907
_GL_HALF_FLOAT = 0x140B

_gl = None          # the loaded GL library; False once loading has failed
_ENABLED = True     # set False to force the plain RGB16F path everywhere


def _load_gl():
    global _gl
    if _gl is not None:
        return _gl or None
    try:
        if sys.platform == "win32":
            lib = ctypes.WinDLL("opengl32")
        elif sys.platform == "darwin":
            lib = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
        else:
            lib = ctypes.CDLL("libGL.so.1")
        lib.glTexImage2D.argtypes = [
            ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
        ]
        lib.glGenTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        lib.glDeleteTextures.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        lib.glBindTexture.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glPixelStorei.argtypes = [ctypes.c_uint, ctypes.c_int]
        lib.glGetError.restype = ctypes.c_uint
        _gl = lib
    except Exception as e:
        print(f"[packed_lightmap] GL library unavailable ({e}) - using RGB16F lightmaps.")
        _gl = False
    return _gl or None


class PackedLightmapTexture:
    """A moderngl texture whose GL object this class owns (see module docstring). Anything not
    defined here (use, size, filter, ...) is forwarded to the wrapped moderngl texture."""

    def __init__(self, texture, glo):
        self._texture = texture
        self._glo = glo

    def __getattr__(self, name):
        return getattr(self._texture, name)

    def release(self):
        glo, self._glo = self._glo, 0
        if glo:
            try:
                self._texture.release()
            finally:
                _gl.glDeleteTextures(1, ctypes.byref(ctypes.c_uint(glo)))


def create_packed_lightmap(ctx, array):
    """array: (H, W, 3) float16, non-negative. Returns a PackedLightmapTexture, or None if the
    packed format couldn't be used (the caller then makes a normal RGB16F texture)."""
    gl = _load_gl() if _ENABLED else None
    if gl is None:
        return None
    height, width = array.shape[:2]
    data = array.tobytes()
    glo = ctypes.c_uint(0)
    try:
        gl.glGetError()                                  # clear any stale error
        gl.glGenTextures(1, ctypes.byref(glo))
        gl.glBindTexture(_GL_TEXTURE_2D, glo.value)
        gl.glPixelStorei(_GL_UNPACK_ALIGNMENT, 1)        # rows are 6*width bytes, not nec. 4-aligned
        gl.glTexImage2D(_GL_TEXTURE_2D, 0, _GL_R11F_G11F_B10F, width, height, 0,
                        _GL_RGB, _GL_HALF_FLOAT, data)
        gl.glPixelStorei(_GL_UNPACK_ALIGNMENT, 4)
        if gl.glGetError() != 0:
            raise RuntimeError("GL error uploading R11F_G11F_B10F texture")
        texture = ctx.external_texture(glo.value, (width, height), 3, 0, "f2")
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)   # also makes it mip-complete
        return PackedLightmapTexture(texture, glo.value)
    except Exception as e:
        print(f"[packed_lightmap] packed upload failed ({e}) - using RGB16F for this lightmap.")
        if glo.value:
            gl.glDeleteTextures(1, ctypes.byref(glo))
        return None
