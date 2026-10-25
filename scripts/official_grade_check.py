"""Grade the gold patch and the empty patch with the OFFICIAL swegemma Phase 2 rules, offline.

Reproduces swegemma/harness/verification.py step by step, using the harness's own code where it is importable
(setup_workspace_test_config's script, _validate_junit_xml), on a Berth snapshot with the task's offline environment:
  1. snapshot -> workspace; offline venv from the wheel pool (as run_gemma.py); write the harness pytest.ini/conftest;
     baseline commit
  2. apply the candidate patch (gold or empty); reset the target test files to HEAD; apply test_patch
  3. rewrite the harness pytest.ini/conftest (overwrite=True)
  4. run the harness's exact pytest command (python3 -s -m pytest <files> --junitxml=... -p no:anyio -o timeout=0
     -o python_classes="Test* *Test" -q, with PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1)
  5. resolved = exit code 0 AND _validate_junit_xml(...) passes AND the stdout summary shows passing tests
A task is harness-compatible when gold -> resolved and empty -> not resolved.

Must run with a Python that has swegemma installed (comp_data/venv_adk). Writes data/official_grade/<id>.json.

usage: unshare -rn comp_data/venv_adk/bin/python -I scripts/official_grade_check.py --release release \
          --python <py3.13> --uv <uv> --wheels release/wheels --workers 2 [--split core]
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gemma as rg  # noqa: E402  (checkout, make_env, CFG)
from swegemma.harness import container_setup as cs  # noqa: E402
from swegemma.harness.verification import _validate_junit_xml  # noqa: E402
from swegemma.models.task import extract_test_files_from_patch  # noqa: E402


class LocalExec:
    """Minimal stand-in for the sandbox manager: runs the harness's own setup commands locally."""

    def exec(self, _cid, cmd, timeout=None):
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout or 300)
        return type("R", (), {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr})()


def sh(cmd, cwd, env=None, timeout=900):
    """Run in a new process group and kill the whole group on timeout (tests can spawn servers that keep pipes open)."""
    import signal
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         shell=isinstance(cmd, str), executable="/bin/bash" if isinstance(cmd, str) else None,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        try:
            out, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return 124, out or "", (err or "") + "\ntimeout"


def prune_for_validator(xml, f2p, p2p, test_patch):
    """The official validator matches required tests by name PREFIX, so a required test fails if any prefix-matching
    sibling (e.g. test_x_windows for test_x) is skipped or failed. Drop such colliding PASS_TO_PASS ids, and
    colliding FAIL_TO_PASS ids while at least one FAIL_TO_PASS remains. Returns (f2p, p2p, dropped_f2p, dropped_p2p)."""
    f2p, p2p, df, dp = list(f2p), list(p2p), [], []
    for _ in range(len(f2p) + len(p2p) + 1):
        ok, reason = _validate_junit_xml(xml, fail_to_pass=tuple(f2p), pass_to_pass=tuple(p2p), test_patch=test_patch)
        if ok or not reason or not reason.startswith("Required test node did not pass: "):
            return f2p, p2p, df, dp, ok
        bad = reason.split(": ", 1)[1]
        if bad in p2p:
            p2p.remove(bad); dp.append(bad)
        elif bad in f2p and len(f2p) > 1:
            f2p.remove(bad); df.append(bad)
        else:
            return f2p, p2p, df, dp, False
    return f2p, p2p, df, dp, False


def grade(task, patch, release, cache, label, prune=False):
    work = Path(tempfile.mkdtemp(prefix=f"og_{task['instance_id']}_"))
    ws = work / "workspace"
    try:
        rg.checkout(release, task, ws)
        venv = rg.make_env(task, ws, cache)
        env = dict(os.environ, PATH=f"{venv}/bin:" + os.environ["PATH"], VIRTUAL_ENV=str(venv))
        # .venv must not show up in diffs
        with open(ws / ".git/info/exclude", "a") as fh:
            fh.write("\n.venv/\n__pycache__/\n*.pyc\n.pytest_cache/\n*.egg-info/\n")
        cs.setup_workspace_test_config(LocalExec(), "local", repo=task["repo"], workspace_dir=str(ws))
        sh("git add -A && git -c user.email=a@b -c user.name=eval commit -q -m eval_baseline --allow-empty", ws)
        if patch.strip():
            (work / "agent.patch").write_text(patch)
            rc, _, err = sh(["git", "apply", "--whitespace=nowarn", str(work / "agent.patch")], ws)
            if rc:
                return {"resolved": False, "error": f"agent patch did not apply: {err[-300:]}"}
        files = extract_test_files_from_patch(task["test_patch"])
        for f in files:  # reset target test files (tracked -> HEAD, untracked -> removed)
            sh(f"git checkout HEAD -- {shlex.quote(f)} 2>/dev/null; git clean -fq -- {shlex.quote(f)}", ws)
        (work / "test.patch").write_text(task["test_patch"])
        rc, _, err = sh(["git", "apply", "--whitespace=nowarn", str(work / "test.patch")], ws)
        if rc:
            return {"resolved": False, "error": f"test patch did not apply: {err[-300:]}"}
        cs.setup_workspace_test_config(LocalExec(), "local", repo=task["repo"], workspace_dir=str(ws), overwrite=True)
        targets = [p for p in files if Path(p).name != "conftest.py" and p.endswith(".py")] or files
        junit = work / "junit.xml"
        cmd = (f"cd {shlex.quote(str(ws))} && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest "
               f"{' '.join(shlex.quote(p) for p in targets) or '.'} --junitxml={junit} "
               "-p no:anyio -o timeout=0 -o python_classes=\"Test* *Test\" -q")
        rc, out, err = sh(cmd, ws, env=env, timeout=300)
        resolved, why = rc == 0, None
        if resolved:
            xml = junit.read_text() if junit.exists() else ""
            stdout = out.lower()
            stdout_invalid = (bool(re.search(r"\b0 passed\b", stdout)) or "no tests ran" in stdout
                              or "collected 0 items" in stdout or (stdout.strip() != "" and "passed" not in stdout))
            if xml.strip():
                pruned = None
                if prune:
                    f2p, p2p, df, dp, _ = prune_for_validator(xml, task["FAIL_TO_PASS"], task["PASS_TO_PASS"],
                                                              task["test_patch"])
                    pruned = {"FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p, "dropped_f2p": df, "dropped_p2p": dp}
                    task = dict(task, FAIL_TO_PASS=f2p, PASS_TO_PASS=p2p)
                ok, reason = _validate_junit_xml(xml, fail_to_pass=tuple(task["FAIL_TO_PASS"]),
                                                 pass_to_pass=tuple(task["PASS_TO_PASS"]), test_patch=task["test_patch"])
                if not ok:
                    resolved, why = False, reason
                elif stdout_invalid:
                    resolved, why = False, "stdout shows no passing tests"
            else:
                resolved, why = False, "missing junit xml"
        tail = [l for l in out.strip().splitlines() if l.strip()][-1:] if out.strip() else []
        failed = re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)[:20]
        res = {"resolved": resolved, "exit_code": rc, "reason": why, "summary": tail[0] if tail else "",
               "failed": failed, "output_tail": (out + "\n" + err)[-1500:]}
        if prune and rc == 0 and junit.exists():
            res["pruned"] = locals().get("pruned")
        return res
    except Exception as e:  # noqa: BLE001
        return {"resolved": False, "error": f"{type(e).__name__}: {e}"[:400]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--python", required=True)
    ap.add_argument("--uv", default="uv")
    ap.add_argument("--wheels", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/home/claude/uvcache_og"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--split", nargs="*", default=["core", "extended", "overlap"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--out", type=Path, default=Path("data/official_grade"))
    a = ap.parse_args()
    rg.CFG.update(uv=a.uv, python=a.python, wheels=str(a.wheels.resolve()))
    a.release = a.release.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in (a.release / "tasks.jsonl").read_text().splitlines() if l.strip()]
    tasks = [t for t in tasks if t["split"] in a.split and (not a.tasks or t["instance_id"] in a.tasks)]

    def one(t):
        f = a.out / f"{t['instance_id']}.json"
        if f.exists():
            return json.loads(f.read_text())
        t0 = time.time()
        g = grade(t, t["patch"], a.release, a.cache, "gold", prune=True)
        pr = g.get("pruned") or {}
        te = dict(t, FAIL_TO_PASS=pr.get("FAIL_TO_PASS", t["FAIL_TO_PASS"]), PASS_TO_PASS=pr.get("PASS_TO_PASS", t["PASS_TO_PASS"]))
        e = grade(te, "", a.release, a.cache, "empty") if g.get("resolved") else {"resolved": None, "skipped": "gold failed"}
        rec = {"instance_id": t["instance_id"], "split": t["split"], "gold": g, "empty": e,
               "compatible": bool(g.get("resolved")) and not e.get("resolved"), "seconds": round(time.time() - t0, 1)}
        f.write_text(json.dumps(rec, indent=1))
        print(f"{t['instance_id']}: compatible={rec['compatible']} gold={g.get('resolved')} empty={e.get('resolved')} "
              f"{(g.get('reason') or g.get('error') or g.get('summary') or '')[:120]}", flush=True)
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, tasks))
    from collections import Counter
    print(Counter((r["split"], r["compatible"]) for r in res))


if __name__ == "__main__":
    main()
