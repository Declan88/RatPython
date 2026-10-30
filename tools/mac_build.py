"""
Runs the GitHub macOS build (.github/workflows/build-macos.yml) for "BuildCMD_Nuitka_Standalone_Packed.cmd":

    python tools/mac_build.py start [0.1.5]         # trigger the workflow on master, remember its run id
    python tools/mac_build.py fetch C:\\Steambuilds\\ContentMac   # wait for it, unzip RatWar.app there

GitHub builds what is PUSHED to master, not this working tree - `start` says so (and asks) when local
changes or commits aren't on origin/master yet, since the Mac build would then differ from the Windows one.
The game version is the exception: `start` hands it to the workflow (which writes it into Modules/version.py
before compiling), so a version just bumped for this build needn't be pushed first.
Needs the GitHub CLI (gh), logged in. Exits 1 on any failure.
"""

import json
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = "build-macos.yml"
BRANCH = "master"
RUN_FILE = ROOT / "build_nuitka" / "mac_run.txt"
VERSION_FILE = "Modules/version.py"
DOWNLOAD_DIR = ROOT / "build_nuitka" / "mac_artifact"


def _run(*args, capture=True):
    result = subprocess.run(args, cwd=ROOT, capture_output=capture, text=True)
    if result.returncode != 0:
        sys.exit(f"{' '.join(args)} failed:\n{(result.stderr or '').strip()}")
    return result.stdout if capture else ""


def _latest_run_id():
    runs = json.loads(_run("gh", "run", "list", "--workflow", WORKFLOW, "--event", "workflow_dispatch",
                           "--limit", "1", "--json", "databaseId"))
    return runs[0]["databaseId"] if runs else 0


def _check_pushed():
    """Lists what the Mac build WON'T include and asks whether to go on anyway."""
    _run("git", "fetch", "--quiet", "origin", BRANCH)
    unpushed = _run("git", "log", "--oneline", f"origin/{BRANCH}..HEAD").splitlines()
    # .pyc files are tracked in this repo but never reach a build - not worth a warning. Nor is the version file:
    # the version goes to the workflow directly (see start).
    dirty = [line for line in _run("git", "status", "--porcelain").splitlines()
             if not line.endswith(".pyc") and not line.endswith(VERSION_FILE)]
    if not unpushed and not dirty:
        return
    print(f"WARNING: the Mac build uses origin/{BRANCH}, which is missing:")
    if unpushed:
        print(f"  {len(unpushed)} unpushed commit(s), e.g. {unpushed[0]}")
    if dirty:
        print(f"  {len(dirty)} uncommitted change(s), e.g.")
        for line in dirty[:8]:
            print(f"    {line}")
    try:
        answer = input("Build the Mac version without them anyway? [y/N] ")
    except EOFError:        # no console to answer from (run unattended)
        answer = ""
    if answer.strip().lower() != "y":
        sys.exit("Stopped - commit and push first, or run with --no-mac.")


def start(version=None):
    _check_pushed()
    before = _latest_run_id()
    extra = ("-f", f"version={version}") if version else ()
    _run("gh", "workflow", "run", WORKFLOW, "--ref", BRANCH, *extra)
    # The dispatch returns before the run exists; wait for a run newer than the one before it.
    for _ in range(30):
        time.sleep(2)
        run_id = _latest_run_id()
        if run_id and run_id != before:
            break
    else:
        sys.exit(f"Triggered {WORKFLOW}, but no new run showed up - check the repository's Actions tab.")
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_FILE.write_text(str(run_id))
    print(f"macOS build started: run {run_id}")


def _clear(dest):
    """Empties dest, refusing if it holds anything but .app bundles (a mistyped path must not be wiped)."""
    if not dest.exists():
        dest.mkdir(parents=True)
        return
    other = [p.name for p in dest.iterdir() if not (p.is_dir() and p.suffix == ".app")]
    if other:
        sys.exit(f"{dest} holds more than .app bundles ({', '.join(other[:5])}) - not wiping it.")
    for app in dest.iterdir():
        shutil.rmtree(app)


def fetch(dest):
    if not RUN_FILE.is_file():
        sys.exit(f"{RUN_FILE} missing - run `python tools/mac_build.py start` first.")
    run_id = RUN_FILE.read_text().strip()
    print(f"Waiting for macOS build run {run_id} ...")
    _run("gh", "run", "watch", run_id, "--exit-status", "--interval", "30", capture=False)

    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
    _run("gh", "run", "download", run_id, "--dir", str(DOWNLOAD_DIR))
    zips = list(DOWNLOAD_DIR.rglob("*.zip"))
    if len(zips) != 1:
        sys.exit(f"Expected one zip in the run's artifacts, found {len(zips)}.")

    with zipfile.ZipFile(zips[0]) as z:
        infos = z.infolist()
        # Windows can't store symlinks or Unix permission bits, so steamcmd uploads neither from here.
        # Today's bundle has no symlinks and one executable (the game binary); stop if that changes.
        links = [i.filename for i in infos if stat.S_ISLNK(i.external_attr >> 16)]
        if links:
            sys.exit(f"The Mac zip holds {len(links)} symlink(s), e.g. {links[0]} - can't upload those from Windows.")
        apps = {Path(i.filename).parts[0] for i in infos}
        if len(apps) != 1 or not next(iter(apps)).endswith(".app"):
            sys.exit(f"Expected one .app bundle at the zip's top level, found {sorted(apps)}.")
        _clear(dest)
        z.extractall(dest)
    execs = [i.filename for i in infos if not i.is_dir() and (i.external_attr >> 16) & 0o111]
    print(f"macOS build unpacked into {dest / next(iter(apps))} ({len(infos)} entries, executable: {execs})")


def main():
    if len(sys.argv) in (2, 3) and sys.argv[1] == "start":
        start(sys.argv[2] if len(sys.argv) == 3 else None)
    elif len(sys.argv) == 3 and sys.argv[1] == "fetch":
        fetch(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
