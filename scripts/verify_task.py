"""Stage 2: verify one candidate commit as a SWE-style task under Python 3.13.

For candidate (repo, parent, sha):
  1. check out `parent` in a fresh worktree;
  2. split the commit's diff into test_patch (test files) and patch (everything else);
  3. build a Python 3.13 venv with the repo and its test dependencies (uv, cached);
  4. run the touched test files on parent+test_patch (state A) and on parent+test_patch+patch (state B, twice);
  5. FAIL_TO_PASS = tests that pass in both B runs but did not pass in A;
     PASS_TO_PASS = tests that pass in A and in both B runs.
     The task is kept when FAIL_TO_PASS is non-empty and no test that passed in A fails in B.

Writes <out_dir>/<instance_id>.json with the task row, the verdict, timings and the environment's frozen package list.

usage: python -I verify_task.py <candidates.jsonl> <line_no> --repos repos --work /scratch/work --out data/verify
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

PY = "3.13"
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
GROUPS = ("tests", "test", "testing", "dev")
TEST_TIMEOUT = 600


def sh(cmd, cwd=None, timeout=900, env=None, check=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    if check and r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))} -> {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def split_patch(repo, parent, sha):
    files = sh(["git", "-C", repo, "diff", "--name-only", parent, sha], check=True).stdout.split()
    tests = [f for f in files if TEST_RE.search(f)]
    rest = [f for f in files if f not in tests]
    diff = lambda paths: sh(["git", "-C", repo, "diff", "--binary", parent, sha, "--", *paths], check=True).stdout if paths else ""
    return diff(rest), diff(tests), rest, tests


def install_plan(ws: Path):
    """Return (description, list of uv pip install arg lists)."""
    pp = ws / "pyproject.toml"
    data = tomllib.loads(pp.read_text()) if pp.exists() else {}
    groups = data.get("dependency-groups", {}) or {}
    extras = (data.get("project", {}) or {}).get("optional-dependencies", {}) or {}
    for g in GROUPS:
        if g in groups:
            return f"group:{g}", [["-e", ".", "--group", g, "pytest"]]
    for g in GROUPS:
        if g in extras:
            return f"extra:{g}", [["-e", f".[{g}]", "pytest"]]
    reqs = sorted([p for pat in ("requirements/test*.txt", "requirements/tests*.txt", "requirements-test*.txt",
                                 "test-requirements.txt", "requirements/dev*.txt", "requirements-dev*.txt",
                                 "requirements_dev*.txt", "tests/requirements*.txt") for p in ws.glob(pat)])
    if reqs:
        return f"req:{reqs[0].relative_to(ws)}", [["-r", str(reqs[0]), "pytest"], ["-e", ".", "--no-deps"]]
    return "plain", [["-e", ".", "pytest"]]


def run_tests(ws, py, test_files, tag):
    xml = ws / f".junit_{tag}.xml"
    t0 = time.time()
    r = sh([py, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "-rN", "--tb=no",
            f"--junitxml={xml}", *test_files], cwd=ws, timeout=TEST_TIMEOUT)
    out = {}
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            name = f"{tc.get('classname', '')}::{tc.get('name')}"
            kids = {c.tag for c in tc}
            out[name] = ("failed" if kids & {"failure", "error"} else "skipped" if "skipped" in kids else "passed")
        xml.unlink()
    return out, r.returncode, round(time.time() - t0, 1), r.stdout[-1500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", type=Path)
    ap.add_argument("line", type=int)
    ap.add_argument("--repos", type=Path, default=Path("repos"))
    ap.add_argument("--work", type=Path, default=Path("/home/claude/work"))
    ap.add_argument("--out", type=Path, default=Path("data/verify"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pin-to-commit-date", action="store_true",
                    help="resolve dependencies as of the fix commit's date (uv --exclude-newer); used when the newest "
                         "releases break tests at an older commit")
    ap.add_argument("--force", action="store_true", help="re-verify even if a record exists")
    a = ap.parse_args()
    c = json.loads(a.candidates.read_text().splitlines()[a.line])
    name = c["repo"].split("/")[1]
    iid = f"{name}_{c['pr']}"
    a.out.mkdir(parents=True, exist_ok=True)
    out_file = a.out / f"{iid}.json"
    if out_file.exists() and not a.force:
        return
    repo = str((a.repos / c["repo"].replace("/", "_")).resolve())
    ws = (a.work / iid).resolve()
    rec = {"instance_id": iid, "repo": c["repo"], "base_commit": c["parent"], "fix_commit": c["sha"], "pr": c["pr"],
           "created_at": c["date"], "subject": c["subject"], "status": "error", "t": {}}
    t0 = time.time()
    try:
        if ws.exists():
            sh(["git", "-C", repo, "worktree", "remove", "--force", str(ws)])
            shutil.rmtree(ws, ignore_errors=True)
        sh(["git", "-C", repo, "worktree", "add", "--detach", str(ws), c["parent"]], check=True, timeout=1800)
        patch, test_patch, src_files, test_files = split_patch(repo, c["parent"], c["sha"])
        rec.update(patch=patch, test_patch=test_patch, patch_files=src_files, test_patch_files=test_files)
        py_tests = [f for f in test_files if f.endswith(".py") and not f.endswith("conftest.py")]
        rec["t"]["checkout"] = round(time.time() - t0, 1)

        # environment
        t1 = time.time()
        sh(["uv", "venv", "-q", "-p", PY, str(ws / ".venv")], check=True)
        env = dict(os.environ, VIRTUAL_ENV=str(ws / ".venv"))
        desc, steps = install_plan(ws)
        rec["install"] = desc
        pin = ["--exclude-newer", c["date"]] if a.pin_to_commit_date else []
        rec["env_pinned_to_commit_date"] = bool(pin)
        for step in steps:
            r = sh(["uv", "pip", "install", "-q", "-p", str(ws / ".venv/bin/python"), *pin, *step], cwd=ws, env=env)
            if r.returncode:
                rec.update(status="install_failed", log=r.stderr[-3000:])
                raise SystemExit
        py = str(ws / ".venv/bin/python")
        rec["t"]["install"] = round(time.time() - t1, 1)

        # state A: parent + test_patch
        (ws / ".test.patch").write_text(test_patch)
        sh(["git", "apply", ".test.patch"], cwd=ws, check=True)
        A, rcA, tA, logA = run_tests(ws, py, py_tests, "A")
        # state B: + gold patch
        (ws / ".gold.patch").write_text(patch)
        sh(["git", "apply", ".gold.patch"], cwd=ws, check=True)
        B1, rcB, tB, logB = run_tests(ws, py, py_tests, "B1")
        B2, _, _, _ = run_tests(ws, py, py_tests, "B2")
        rec["t"].update(tests_A=tA, tests_B=tB)
        passB = {k for k in B1 if B1[k] == "passed" and B2.get(k) == "passed"}
        f2p = sorted(k for k in passB if A.get(k) != "passed")
        p2p = sorted(k for k in passB if A.get(k) == "passed")
        broke = sorted(k for k, v in A.items() if v == "passed" and k not in passB)
        flaky = sorted(k for k in B1 if (B1[k] == "passed") != (B2.get(k) == "passed"))
        rec.update(FAIL_TO_PASS=f2p, PASS_TO_PASS=p2p, broke=broke, flaky=flaky, n_tests_B=len(B1))
        if not B1:
            rec.update(status="no_tests", log=logB)
        elif broke or flaky:
            rec["status"] = "unstable"
        elif not f2p:
            rec["status"] = "no_f2p"
        else:
            rec["status"] = "ok"
        rec["freeze"] = sh(["uv", "pip", "freeze", "-p", py], env=env).stdout.split("\n")
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", log=f"{type(e).__name__}: {e}"[-3000:])
    finally:
        rec["t"]["total"] = round(time.time() - t0, 1)
        out_file.write_text(json.dumps(rec, indent=1))
        if not a.keep:
            sh(["git", "-C", repo, "worktree", "remove", "--force", str(ws)])
            shutil.rmtree(ws, ignore_errors=True)
    print(iid, rec["status"], rec["t"].get("total"), len(rec.get("FAIL_TO_PASS", [])), file=sys.stderr)


if __name__ == "__main__":
    main()
