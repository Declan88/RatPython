"""
Sanity-checks a finished packed standalone build (see "BuildCMD_Nuitka_Standalone_Packed.cmd"):

    python tools/check_packed_build.py [build_nuitka/app.dist] [RatWar.exe]

  * the Assets folder holds ONLY the pack (pack.idx + pk/*.dat) - no plain PNG/GLB/WAV/EXR/... leaked in
  * every asset in the index has its blob, and the index decrypts with the key the build was made with
  * no .py/.pyc source ships
  * the game exe exports NvOptimusEnablement and AmdPowerXpressRequestHighPerformance (needs `pip install
    pefile`), so switchable-graphics laptops use the dedicated GPU

Exits 1 on any failure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Modules.Assets import pack_format as fmt        # noqa: E402


def main():
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build_nuitka" / "app.dist")
    exe_name = sys.argv[2] if len(sys.argv) > 2 else "RatWar.exe"
    problems = []

    assets = dist / "Assets"
    if not (assets / fmt.INDEX_NAME).is_file():
        sys.exit(f"{assets / fmt.INDEX_NAME} missing - the pack was not copied into the build")
    stray = [p for p in assets.rglob("*") if p.is_file()
             and p.relative_to(assets).as_posix() != fmt.INDEX_NAME
             and not p.relative_to(assets).as_posix().startswith(fmt.BLOB_DIR + "/")]
    if stray:
        problems.append(f"{len(stray)} plain files in Assets (should be none), e.g. {stray[0].relative_to(dist)}")

    try:
        from Modules.Assets._asset_key import KEY
        index = fmt.decode_index((assets / fmt.INDEX_NAME).read_bytes(), KEY)
        missing = [rel for rel, (_, name, _, _) in index.items() if not (assets / fmt.BLOB_DIR / name).is_file()]
        if missing:
            problems.append(f"{len(missing)} indexed assets have no blob, e.g. {missing[0]}")
        print(f"pack: {len(index)} assets indexed, {len(missing)} missing blobs")
    except Exception as e:
        problems.append(f"index could not be read with this key: {e}")

    sources = [p for p in dist.rglob("*") if p.suffix in (".py", ".pyc")]
    if sources:
        problems.append(f"{len(sources)} Python source/bytecode files ship, e.g. {sources[0].relative_to(dist)}")
    print(f"python source files in the build: {len(sources)}")

    exe = dist / exe_name
    if not exe.is_file():
        problems.append(f"{exe_name} not found in {dist}")
    else:
        try:
            import pefile
            pe = pefile.PE(str(exe), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
            names = {s.name.decode() for s in getattr(pe.DIRECTORY_ENTRY_EXPORT, "symbols", []) if s.name} \
                if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else set()
            need = {"NvOptimusEnablement", "AmdPowerXpressRequestHighPerformance"}
            print(f"{exe_name} GPU-preference exports: {sorted(need & names)}")
            if need - names:
                problems.append(f"{exe_name} is missing exports {sorted(need - names)} (laptops may pick the integrated GPU)")
        except ImportError:
            print("pefile not installed - skipped the GPU export check (pip install pefile)")

    # The key must not sit in the exe in a form `strings` / a hex editor search finds.
    try:
        from Modules.Assets._asset_key import KEY
        exe_bytes = exe.read_bytes() if exe.is_file() else b""
        found = [label for label, needle in (("raw bytes", KEY), ("hex text", KEY.hex().encode()),
                                              ("hex text (UTF-16)", KEY.hex().encode("utf-16-le")))
                 if needle in exe_bytes]
        print(f"decryption key readable in {exe_name}: {found or 'no'}")
        if found:
            problems.append(f"the asset key is readable in {exe_name} as {', '.join(found)} - rebuild with the "
                            f"current tools/pack_assets.py (it writes the key module in a non-literal form)")
    except ImportError:
        pass

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()

