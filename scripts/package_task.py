"""Stage 5: package a verified task in the competition layout and re-verify it OFFLINE.

Per task (status ok):
  snapshots/<iid>.tgz   repo at base_commit as a fresh one-commit git repo (no forward history, like the official
                        snapshots), so `git status/diff` work and the fix cannot leak through `git log`
  graphs/<name>_<base>.json      official-format code graph (scripts/third_party/code_graph.py)
  embeddings/<name>_<base>.npz   stand-in vectors (scripts/third_party/random_embeddings.py); see dataset card
  wheels/                        every distribution the task's tests need, incl. build backends (shared pool)
Then, inside a network namespace with no network (unshare -rn): fresh Python 3.13 venv, install from wheels/ only,
apply test_patch -> FAIL_TO_PASS must fail and PASS_TO_PASS pass; apply patch -> all must pass; empty patch -> F2P
fails. Results go into the record under "offline".

usage: python -I package_task.py <verify dir> <instance_id> --release release --repos repos
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))
import code_graph  # noqa: E402
import random_embeddings  # noqa: E402

PY = "3.13"
WHEEL_PY = ["--python-version", "3.13", "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64",
            "--platform", "linux_x86_64", "--implementation", "cp", "--abi", "cp313", "--abi", "abi3", "--abi", "none"]


def sh(cmd, cwd=None, timeout=900, env=None, check=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))} -> {r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


def requirements(rec, ws):
    """(pinned requirement lines from the verified env minus the task's own package, build-backend requirements)."""
    reqs = [l for l in rec.get("freeze", []) if l and not l.startswith("-e") and " @ file://" not in l]
    pp = ws / "pyproject.toml"
    build = tomllib.loads(pp.read_text()).get("build-system", {}).get("requires", []) if pp.exists() else []
    if any(b.lower().startswith("hatchling") for b in build):
        build.append("editables")  # hatchling's editable install imports it without declaring it
    return sorted(set(reqs)), sorted(set(build))


def fetch_wheels(reqs, build, wheels: Path):
    wheels.mkdir(parents=True, exist_ok=True)
    pip = [sys.executable, "-m", "pip", "-q", "--disable-pip-version-check"]
    missing = []
    if build:  # build backends with their whole dependency closure
        r = sh(pip + ["download", "--only-binary=:all:", *WHEEL_PY, "-d", str(wheels), *build], timeout=900)
        if r.returncode:
            missing += build
    for req in reqs:
        r = sh(pip + ["download", "--no-deps", "--only-binary=:all:", *WHEEL_PY, "-d", str(wheels), req], timeout=600)
        if r.returncode:  # sdist only: build a wheel with 3.13
            r = sh(["uv", "run", "-q", "--no-project", "-p", PY, "--with", "pip", "python", "-m", "pip", "-q", "wheel",
                    "--no-deps", "-w", str(wheels), req], timeout=900)
            if r.returncode:
                missing.append(req)
    return missing


def run_tests(ws, py, files, tag, netless=True):
    xml = ws / f".junit_{tag}.xml"
    cmd = [py, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=no", f"--junitxml={xml}", *files]
    if netless:
        cmd = ["unshare", "-rn"] + cmd
    sh(cmd, cwd=ws, timeout=900)
    out = {}
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            kids = {c.tag for c in tc}
            out[f"{tc.get('classname', '')}::{tc.get('name')}"] = (
                "failed" if kids & {"failure", "error"} else "skipped" if "skipped" in kids else "passed")
        xml.unlink()
    return out


def make_snapshot(repo, base, dest: Path, snap: Path):
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    a = subprocess.Popen(["git", "-C", repo, "archive", base], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=a.stdout, check=True)
    a.wait()
    env = dict(os.environ, GIT_AUTHOR_NAME="snapshot", GIT_AUTHOR_EMAIL="snapshot@localhost",
               GIT_COMMITTER_NAME="snapshot", GIT_COMMITTER_EMAIL="snapshot@localhost",
               GIT_AUTHOR_DATE="2000-01-01T00:00:00Z", GIT_COMMITTER_DATE="2000-01-01T00:00:00Z")
    for c in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A", "-f"], ["git", "commit", "-q", "-m", f"base {base}"]):
        sh(c, cwd=dest, env=env, check=True)
    snap.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snap, "w:gz") as t:
        t.add(dest, arcname=".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_dir", type=Path)
    ap.add_argument("iid")
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--repos", type=Path, default=Path("repos"))
    ap.add_argument("--work", type=Path, default=Path("/home/claude/work_pkg"))
    a = ap.parse_args()
    f = a.verify_dir / f"{a.iid}.json"
    rec = json.loads(f.read_text())
    if rec.get("status") != "ok" or rec.get("offline", {}).get("status") in ("ok", "fail"):
        return
    t0 = time.time()
    repo = str((a.repos / rec["repo"].replace("/", "_")).resolve())
    name, base = rec["repo"].split("/")[1], rec["base_commit"]
    ws = (a.work / a.iid).resolve()
    off = {"status": "error"}
    try:
        make_snapshot(repo, base, ws, a.release / "snapshots" / f"{a.iid}.tgz")
        # graph + stand-in embeddings (one per base commit, shared by tasks on the same commit)
        gstem = f"{name}_{base}"
        gpath = a.release / "graphs" / f"{gstem}.json"
        if not gpath.exists():
            gpath.parent.mkdir(parents=True, exist_ok=True)
            nodes, links, _ = code_graph.build(ws, seed=gstem)
            gpath.write_text(code_graph.to_json(code_graph.to_graph(gstem, nodes, links)))
        epath = a.release / "embeddings" / f"{gstem}.npz"
        if not epath.exists():
            epath.parent.mkdir(parents=True, exist_ok=True)
            import numpy as np
            g = json.loads(gpath.read_text())["nodes"]
            np.savez_compressed(epath, **{n["id"]: random_embeddings.node_vector(n["id"], n.get("text") or "") for n in g})
        # wheels
        reqs, build = requirements(rec, ws)
        missing = fetch_wheels(reqs, build, a.release / "wheels")
        off["requirements"], off["build_requires"] = reqs, build
        if missing:
            off.update(status="wheels_missing", missing=missing)
            raise SystemExit
        # offline env
        wheels = str((a.release / "wheels").resolve())
        netless = ["unshare", "-rn"]
        sh(netless + ["uv", "venv", "-q", "--offline", "-p", PY, str(ws / ".venv")], check=True)
        py = str(ws / ".venv/bin/python")
        req_file = ws / ".offline_reqs.txt"
        req_file.write_text("\n".join(reqs))
        inst = ["uv", "pip", "install", "-q", "--offline", "--no-index", "--find-links", wheels, "-p", py]
        r = sh(netless + inst + ["--no-deps", "-r", str(req_file)], cwd=ws)  # the freeze is the full closure
        if r.returncode == 0:
            r = sh(netless + inst + ["--no-deps", "-e", "."], cwd=ws)  # isolated build env, offline from wheels/
        if r.returncode:
            off.update(status="offline_install_failed", log=r.stderr[-2000:])
            raise SystemExit
        f2p, p2p = rec["FAIL_TO_PASS"], rec["PASS_TO_PASS"]
        tests = [t for t in rec["test_patch_files"] if t.endswith(".py") and not t.endswith("conftest.py")]
        # empty patch (tests only): F2P must not pass
        (ws / ".t.patch").write_text(rec["test_patch"])
        sh(["git", "apply", ".t.patch"], cwd=ws, check=True)
        A = run_tests(ws, py, tests, "A")
        (ws / ".g.patch").write_text(rec["patch"])
        sh(["git", "apply", ".g.patch"], cwd=ws, check=True)
        B = run_tests(ws, py, tests, "B")
        f2p_ok = [t for t in f2p if B.get(t) == "passed"]
        empty_passes = [t for t in f2p if A.get(t) == "passed"]
        p2p_bad = [t for t in p2p if B.get(t) != "passed"]
        off.update(f2p_pass=len(f2p_ok), f2p_total=len(f2p), empty_patch_passes=len(empty_passes), p2p_failed=p2p_bad[:50],
                   n_p2p_failed=len(p2p_bad))
        # tests that need the network or misbehave offline are dropped from P2P; F2P must all hold
        if len(f2p_ok) == len(f2p) and not empty_passes:
            off["status"] = "ok"
            rec["PASS_TO_PASS_offline"] = [t for t in p2p if B.get(t) == "passed"]
        else:
            off["status"] = "fail"
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        off.update(status="error", log=f"{type(e).__name__}: {e}"[-2000:])
    finally:
        off["seconds"] = round(time.time() - t0, 1)
        rec["offline"] = off
        f.write_text(json.dumps(rec, indent=1))
        shutil.rmtree(ws, ignore_errors=True)
    print(a.iid, off["status"], off["seconds"], file=sys.stderr)


if __name__ == "__main__":
    main()
