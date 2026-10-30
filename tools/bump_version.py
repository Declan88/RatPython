"""
The game version in Modules/version.py ("major.minor.patch"):

    python tools/bump_version.py            # raise the patch number by one (0.1.4 -> 0.1.5), print the new version
    python tools/bump_version.py --print    # print it, unchanged
    python tools/bump_version.py --set 0.2.0

Only ever raises the patch number by itself - going to a new minor version, or to 1.0, is a decision made by hand
(--set, or editing the file). BuildCMD_Nuitka_Standalone_Packed.cmd bumps it once per Steam upload, and the macOS
build on GitHub is handed the same number (tools/mac_build.py) so both platforms carry it. Exits 1 on a malformed
version.
"""

import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent.parent / "Modules" / "version.py"
_LINE = re.compile(r'^VERSION = "(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)


def _text():
    with open(VERSION_FILE, encoding="utf-8", newline="") as f:     # newline="": the file's own line endings
        return f.read()


def read():
    match = _LINE.search(_text())
    if match is None:
        sys.exit(f'{VERSION_FILE} has no line like VERSION = "0.1.0"')
    return tuple(int(part) for part in match.groups())


def write(version):
    text = _LINE.sub(f'VERSION = "{version}"', _text(), count=1)     # read before opening for writing empties it
    with open(VERSION_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def main():
    args = sys.argv[1:]
    if args[:1] == ["--print"]:
        print(".".join(map(str, read())))
    elif args[:1] == ["--set"] and len(args) == 2:
        if not re.fullmatch(r"\d+\.\d+\.\d+", args[1]):
            sys.exit(f"not a major.minor.patch version: {args[1]!r}")
        write(args[1])
        print(args[1])
    elif not args:
        major, minor, patch = read()
        version = f"{major}.{minor}.{patch + 1}"
        write(version)
        print(version)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
