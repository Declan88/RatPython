"""
Reads the sprite sheet table out of a Valve .vtf texture (version 7.3+ keeps it
in a "sheet" resource). Source packs several sprite frames into one image and
lists, per "sequence" (what a particle's Sequence Random picks), the frames'
rectangles and how long each is shown - the image itself has no grid to guess
from. Only that table is read here; the pixels come from the exported png.
"""

import struct

_SHEET_TAG = b"\x10\x00\x00"


class Sheet:
    """sequences[n] = list of (display_seconds, (left, top, right, bottom)), the
    rectangle in 0-1 image coordinates measured from the top left."""

    def __init__(self, sequences):
        self.sequences = sequences


def read_vtf_sheet(path):
    """The Sheet in a .vtf file, or None if it has none (or can't be read)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if data[:4] != b"VTF\0" or len(data) < 80:
        return None
    major, minor = struct.unpack_from("<II", data, 4)
    if (major, minor) < (7, 3):
        return None
    resource_count = struct.unpack_from("<I", data, 68)[0]
    offset = None
    for i in range(resource_count):
        base = 80 + i * 8
        tag = data[base:base + 3]
        if tag == _SHEET_TAG:
            offset = struct.unpack_from("<I", data, base + 4)[0]
            break
    if offset is None:
        return None
    try:
        size = struct.unpack_from("<I", data, offset)[0]
        pos = offset + 4
        end = pos + size
        version, count = struct.unpack_from("<II", data, pos)
        pos += 8
        images = 4 if version else 1   # rectangles per frame (one per texture layer)
        sequences = {}
        for _ in range(count):
            number, _clamp, frames, _total = struct.unpack_from("<IIIf", data, pos)
            pos += 16
            entries = []
            for _f in range(frames):
                seconds = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                rects = [struct.unpack_from("<4f", data, pos + 16 * k) for k in range(images)]
                pos += 16 * images
                entries.append((seconds, rects[0]))    # the first layer is the sprite
            sequences[number] = entries
        if pos > end + 4:
            return None
    except struct.error:
        return None
    if not sequences:
        return None
    return Sheet([sequences.get(n, []) for n in range(max(sequences) + 1)])
