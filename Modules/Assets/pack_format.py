"""
The on-disk format of the packed (encrypted) game assets, shared by the packing tool
(tools/pack_assets.py) and the runtime reader (pack_runtime.py).

A packed build ships every asset as its own opaque file instead of plain PNG/GLB/WAV:

    Assets/pack.idx                  the encrypted index: every asset's real path -> its blob
    Assets/pk/<hash>.dat             one encrypted blob per asset

Why one file per asset (and not one big archive): Steam only re-sends the files/chunks of a build that
changed. Every blob here is encrypted DETERMINISTICALLY (the keystream depends only on the key and the
asset's own path), so an asset that didn't change is byte-identical between two builds and costs the
players nothing on an update; only changed assets and the small index are re-downloaded.

Each asset is zlib-compressed first when that helps (GLB models and WAV audio shrink by 15-20%; PNG, EXR
and npz are already compressed and are stored as-is), then encrypted: a stream cipher built from the
standard library alone - SHAKE-256 as a keystream generator (keyed with the secret key + the asset's path)
XORed over the data. There is no authentication tag; a 4-byte magic at the start of every plaintext
(RKP1 = stored as-is, RKP2 = zlib-compressed) only detects a wrong key / corrupt file.

This is obfuscation, not DRM: the key has to be inside the game to decrypt anything, so someone willing to
reverse the program or dump its memory can still recover the assets. It stops browsing the install folder
and generic asset-ripping tools, which is its purpose.
"""

import hashlib
import json
import zlib

import numpy as np

MAGIC = b"RKP1"             # plaintext starts with this when the data is stored as-is
MAGIC_ZLIB = b"RKP2"        # ... or with this when it was zlib-compressed before encryption
_COMPRESS_MIN_GAIN = 0.97   # only keep the compressed form if it is at least ~3% smaller
INDEX_NAME = "pack.idx"
BLOB_DIR = "pk"
BLOB_SUFFIX = ".dat"


def normalize(rel_path):
    """The canonical lookup key for an asset path relative to the Assets folder: forward slashes,
    lower case (Windows paths are case-insensitive)."""
    return rel_path.replace("\\", "/").strip("/").lower()


def blob_name(key, rel_key):
    """The file name an asset's blob is stored under: a keyed hash, so the name reveals nothing about it."""
    return hashlib.sha256(key + b"\x00name\x00" + rel_key.encode("utf-8")).hexdigest()[:32] + BLOB_SUFFIX


def _xor(data, key, tag):
    stream = hashlib.shake_256(key + b"\x00" + tag).digest(len(data))
    return (np.frombuffer(data, dtype=np.uint8) ^ np.frombuffer(stream, dtype=np.uint8)).tobytes()


def encrypt(data, key, tag, compress=True):
    """tag: bytes that make the keystream unique to this item (the asset's normalized path).
    Compresses first (encrypted data can't be compressed afterwards - and Steam's own compression of the
    download therefore couldn't help) unless that doesn't shrink it: PNG/EXR/npz are already compressed."""
    if compress:
        packed = zlib.compress(data, 9)         # deterministic: same input -> same bytes, so unchanged assets stay unchanged
        if len(packed) < len(data) * _COMPRESS_MIN_GAIN:
            return _xor(MAGIC_ZLIB + packed, key, tag)
    return _xor(MAGIC + data, key, tag)


def decrypt(blob, key, tag):
    plain = _xor(blob, key, tag)
    magic = plain[:4]
    if magic == MAGIC:
        return plain[4:]
    if magic == MAGIC_ZLIB:
        return zlib.decompress(plain[4:])
    raise ValueError("asset blob did not decrypt (wrong key, or a corrupt/old file)")


def encode_index(files, key):
    """files: {normalized path: [real relative path, blob name, size, mtime]} -> encrypted bytes."""
    payload = json.dumps({"v": 1, "files": files}, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return encrypt(payload, key, b"index")


def decode_index(blob, key):
    return json.loads(decrypt(blob, key, b"index").decode("utf-8"))["files"]
