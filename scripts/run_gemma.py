"""Run a small harness-like coding agent on the tasks and grade its patches (difficulty labels + trajectories).

The agent sees what the competition agent sees: the issue text and a repo checkout at base_commit, with the same kind
of tools (run_command, read_file, edit_file, write_file, submit_patch). Grading follows the competition: apply the
agent's patch to a fresh checkout, add the task's test_patch (agent edits to test files are discarded), run the
FAIL_TO_PASS and PASS_TO_PASS tests; the task is resolved when all pass.

Models (any OpenAI-compatible chat endpoint with tool calls)
  --endpoint http://127.0.0.1:8000/v1/chat/completions --model <served name>   local vLLM (Kaggle notebook)
  --model gemma-4-31b-it --key-file ~/.gemini_key                              Gemini API (AI Studio), the default
  --model gold               replays the reference fix through the tools (tests the pipeline end to end, no API)

Inputs (from the release folder): tasks.jsonl, snapshots/<id>.tgz. Environments: uv venv per task with the pinned
requirements; online from PyPI, or offline from --wheels (Kaggle, internet off). --python picks the 3.13 interpreter
and --uv the uv binary when they are not on PATH. Outputs: runs/<run>/<id>.json (verdict, patch, usage, timings) and
runs/<run>/<id>.traj.jsonl (every message). Resumable: finished tasks are skipped.

usage: python3 run_gemma.py --release release --run g31_t07_a --model gemma-4-31b-it --key-file ~/.gemini_key \
         [--tasks id1 id2 ...] [--samples 1] [--rpm 15] [--workers 2] [--deadline-min 600]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
MAX_TURNS, MAX_OUT, CMD_TIMEOUT, CTX_CHARS = 40, 6000, 120, 60_000  # 60k chars stays under the 32k-token window
TASK_SECONDS, READ_LINES = 1200, 250  # wall-clock limit per attempt; lines shown by an unranged read_file
CFG = {"uv": "uv", "python": "3.13", "wheels": None}  # set from the command line

SYSTEM = """You are an autonomous software engineer working in a Python repository checked out at /workspace (your \
current directory). Fix the issue described by the user by editing the repository's source code. You cannot ask \
questions. Work efficiently: find the relevant code, reproduce or reason about the problem, make a minimal fix, run \
relevant existing tests if useful, then call submit_patch. Do not modify or add test files; hidden tests will check \
your change. Python 3.13 and the project's dependencies are installed (run `python`, `python -m pytest`)."""

TOOLS = [
    {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in /workspace "
     "(bash). Output is truncated.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                                   "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file, optionally a line range "
     "(1-based, inclusive).", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
     "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace one exact occurrence of old_str "
     "with new_str in a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"},
     "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or overwrite a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "submit_patch", "description": "Finish: submit the current changes "
     "to the repository as your solution.", "parameters": {"type": "object", "properties": {}}}},
]


# ----------------------------------------------------------------------------------------------- environment
def sh(cmd, cwd=None, timeout=900, env=None, shell=False):
    """Run in a new process group; on timeout kill the whole group (children can hold the output pipes open)."""
    import signal
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, shell=shell,
                         executable="/bin/bash" if shell else None, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        try:
            out, _ = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out = ""
        return 124, out or "", f"timed out after {timeout}s"


def checkout(release, task, dest: Path):
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    snap = release / "snapshots" / f"{task['instance_id']}.tgz"
    if not snap.exists():  # Kaggle bundles store snapshots as .snap (Kaggle unpacks .tgz files in datasets)
        snap = snap.with_suffix(".snap")
    with tarfile.open(snap, "r:gz") as t:
        t.extractall(dest, filter="data")


def make_env(task, ws: Path, cache: Path):
    """Python 3.13 venv inside the checkout with the task's pinned requirements and the workspace installed editable
    (offline from CFG["wheels"] when set; uv's cache makes repeats fast)."""
    venv = ws / ".venv"
    uv = CFG["uv"]
    off = ["--offline", "--no-index", "--find-links", CFG["wheels"]] if CFG["wheels"] else []
    rc, _, err = sh([uv, "venv", "-q", "-p", CFG["python"], str(venv)] + (["--offline"] if CFG["wheels"] else []))
    if rc:
        raise RuntimeError(f"venv failed: {err[-500:]}")
    req = ws / ".venv/reqs.txt"
    req.write_text("\n".join(sorted(task["requirements"])))
    env = dict(os.environ, UV_CACHE_DIR=str(cache))
    py = str(venv / "bin/python")
    rc, _, err = sh([uv, "pip", "install", "-q", "--no-deps", "-p", py, *off, "-r", str(req)], env=env)  # full closure
    if rc:
        raise RuntimeError(f"env install failed: {err[-800:]}")
    rc, _, err = sh([uv, "pip", "install", "-q", "--no-deps", "-p", py, *off, "-e", str(ws)], env=env)
    if rc:
        raise RuntimeError(f"editable install failed: {err[-800:]}")
    return venv


def run_env(venv):
    env = dict(os.environ)
    env["PATH"] = f"{venv}/bin:" + env["PATH"]
    env["VIRTUAL_ENV"] = str(venv)
    return env


# ----------------------------------------------------------------------------------------------- tools
def tool_call(name, args, ws: Path, env):
    try:
        if name == "run_command":
            rc, out, err = sh(args["command"], cwd=ws, timeout=CMD_TIMEOUT, env=env, shell=True)
            text = (out + ("\n[stderr]\n" + err if err.strip() else "")).strip() or "(no output)"
            return f"[exit {rc}]\n" + trunc(text)
        p = (ws / args["path"].removeprefix("/workspace/").lstrip("/")).resolve()
        if not str(p).startswith(str(ws.resolve())):
            return "error: path outside the workspace"
        if name == "read_file":
            lines = p.read_text(errors="replace").split("\n")
            s = max(args.get("start_line") or 1, 1)
            e = args.get("end_line") or min(len(lines), s + READ_LINES - 1)
            body = "\n".join(f"{i}\t{l}" for i, l in enumerate(lines[s - 1:e], s))
            if e < len(lines) and not args.get("end_line"):
                body += f"\n[file has {len(lines)} lines; showing {s}-{e}. Use start_line/end_line or grep -n to see more]"
            return trunc(body)
        if name == "edit_file":
            src = p.read_text()
            n = src.count(args["old_str"])
            if n != 1:
                return f"error: old_str found {n} times (must be exactly once)"
            p.write_text(src.replace(args["old_str"], args["new_str"], 1))
            return "ok"
        if name == "write_file":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return "ok"
        return f"error: unknown tool {name}"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"


def trunc(s, n=MAX_OUT):
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars cut]...\n" + s[-n // 2:]


def workspace_patch(ws: Path):
    sh(["git", "add", "-A", "-f", "--", ".", ":(exclude).venv"], cwd=ws)
    _, diff, _ = sh(["git", "diff", "--cached", "--binary"], cwd=ws)
    sh(["git", "reset", "-q"], cwd=ws)
    return diff


# ----------------------------------------------------------------------------------------------- models
class RateLimiter:
    def __init__(self, rpm):
        self.gap, self.lock, self.next = 60.0 / rpm, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0, t - now))


class ChatModel:
    """OpenAI-compatible chat endpoint with tool calls (Gemini API or a local vLLM server)."""

    def __init__(self, model, key, rpm, temperature, endpoint=ENDPOINT, extra=None):
        self.model, self.key, self.rl, self.temperature = model, key, RateLimiter(rpm), temperature
        self.endpoint, self.extra = endpoint, extra or {}

    def __call__(self, messages):
        body = {"model": self.model, "messages": messages, "tools": TOOLS, "temperature": self.temperature,
                **self.extra}
        for attempt in range(8):
            self.rl.wait()
            req = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(), method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read())
                return d["choices"][0]["message"], d.get("usage", {})
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(min(120, 10 * 2 ** attempt))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError):
                time.sleep(15)
        raise RuntimeError("model call failed after retries")


class GoldReplay:
    """Writes the reference fix via write_file calls, then submits (pipeline test)."""

    def __init__(self, task, ws_files):
        self.steps = [{"name": "write_file", "arguments": {"path": p, "content": c}} for p, c in ws_files.items()]
        self.steps.append({"name": "submit_patch", "arguments": {}})

    def __call__(self, messages):
        s = self.steps.pop(0)
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{len(messages)}", "type": "function",
                "function": {"name": s["name"], "arguments": json.dumps(s["arguments"])}}]}, {}


# ----------------------------------------------------------------------------------------------- grading
def grade(release, task, patch, cache, work):
    ws = work / "grade"
    checkout(release, task, ws)
    # agent edits to test files are discarded (as in the competition grader)
    keep = [b for b in re.split(r"(?=^diff --git )", patch, flags=re.M)
            if b.strip() and not TEST_RE.search((re.match(r"diff --git a/(\S+)", b) or [None, ""])[1] or "")]
    result = {"applied": True}
    for name, text in (("agent", "".join(keep)), ("tests", task["test_patch"])):
        if not text.strip():
            continue
        f = ws / f".{name}.patch"
        f.write_text(text)
        rc, _, err = sh(["git", "apply", "--whitespace=nowarn", str(f)], cwd=ws)
        f.unlink()
        if rc:
            result.update(applied=False, apply_error=f"{name}: {err[-300:]}")
            return result | {"resolved": False}
    venv = make_env(task, ws, cache)
    files = [t for t in task["test_patch_files"] if t.endswith(".py") and not t.endswith("conftest.py")]
    xml = ws / ".junit.xml"
    sh([str(venv / "bin/python"), "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", "--tb=no",
        f"--junitxml={xml}", *files], cwd=ws, timeout=900, env=run_env(venv))
    got = {}
    if xml.exists():
        for tc in ET.parse(xml).getroot().iter("testcase"):
            kids = {c.tag for c in tc}
            got[f"{tc.get('classname', '')}::{tc.get('name')}"] = not (kids & {"failure", "error", "skipped"})
    f2p = [t for t in task["FAIL_TO_PASS"] if got.get(t)]
    p2p = [t for t in task["PASS_TO_PASS"] if got.get(t)]
    result.update(f2p_passed=len(f2p), f2p_total=len(task["FAIL_TO_PASS"]), p2p_passed=len(p2p),
                  p2p_total=len(task["PASS_TO_PASS"]),
                  f2p_failed=[t for t in task["FAIL_TO_PASS"] if not got.get(t)][:100],
                  p2p_failed=[t for t in task["PASS_TO_PASS"] if not got.get(t)][:100])
    result["resolved"] = len(f2p) == len(task["FAIL_TO_PASS"]) and len(p2p) == len(task["PASS_TO_PASS"])
    return result


# ----------------------------------------------------------------------------------------------- agent loop
def solve(task, sample, a, model_factory, out_dir: Path, cache: Path):
    tag = f"{task['instance_id']}__s{sample}"
    res_path, traj_path = out_dir / f"{tag}.json", out_dir / f"{tag}.traj.jsonl"
    if res_path.exists():
        return json.loads(res_path.read_text())
    work = Path(tempfile.mkdtemp(prefix=f"g4_{task['instance_id']}_"))
    ws = work / "workspace"
    t0 = time.time()
    rec = {"instance_id": task["instance_id"], "sample": sample, "model": a.model, "temperature": a.temperature}
    try:
        checkout(a.release, task, ws)
        venv = make_env(task, ws, cache)
        env = run_env(venv)
        model = model_factory(task, ws)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Issue to resolve in this repository:\n\n{task['problem_statement']}"}]
        usage, submitted, turns = {"prompt_tokens": 0, "completion_tokens": 0}, False, 0
        with traj_path.open("w") as traj:
            for m in messages:
                traj.write(json.dumps(m) + "\n")
            while turns < MAX_TURNS and not submitted and time.time() - t0 < TASK_SECONDS:
                turns += 1
                left = MAX_TURNS - turns
                if left in (10, 3):  # budget reminder, as a user note before the model's turn
                    note = {"role": "user", "content": f"Note: {left} turns left. Make your fix now if you have not, "
                            "then call submit_patch."}
                    messages.append(note)
                    traj.write(json.dumps(note) + "\n")
                # keep the context bounded: drop the oldest tool outputs' bodies first
                while sum(len(json.dumps(m)) for m in messages) > CTX_CHARS:
                    for m in messages[2:]:
                        if m["role"] == "tool" and not m["content"].startswith("[old output removed"):
                            m["content"] = "[old output removed to save context]"
                            break
                    else:
                        break
                msg, u = model(messages)
                for k in usage:
                    usage[k] += u.get(k, 0) or 0
                msg = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
                msg["role"] = "assistant"
                messages.append(msg)
                traj.write(json.dumps(msg) + "\n")
                calls = msg.get("tool_calls") or []
                if not calls:
                    nudge = {"role": "user", "content": "Continue using the tools. Call submit_patch when done."}
                    messages.append(nudge)
                    traj.write(json.dumps(nudge) + "\n")
                    continue
                for c in calls:
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args, out = {}, "error: arguments are not valid JSON"
                    else:
                        out = "submitted" if name == "submit_patch" else tool_call(name, args, ws, env)
                    if name == "submit_patch":
                        submitted = True
                    tm = {"role": "tool", "tool_call_id": c.get("id", ""), "content": out}
                    messages.append(tm)
                    traj.write(json.dumps(tm) + "\n")
        patch = workspace_patch(ws)
        rec.update(turns=turns, submitted=submitted, usage=usage, patch=patch)
        rec.update(grade(a.release, task, patch, cache, work) if patch.strip() else {"resolved": False, "empty": True})
        rec["status"] = "done"
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:500]}")
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
        shutil.rmtree(work, ignore_errors=True)
    if rec["status"] == "done":
        res_path.write_text(json.dumps(rec, indent=1))
    print(f"{tag}: {rec.get('status')} resolved={rec.get('resolved')} turns={rec.get('turns')} {rec['seconds']}s",
          flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--key-file", type=Path, default=Path.home() / ".gemini_key")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--rpm", type=float, default=15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--max-tokens", type=int, default=None, help="per model call (local servers)")
    ap.add_argument("--thinking", action="store_true", help="send chat_template_kwargs enable_thinking (vLLM)")
    ap.add_argument("--python", default="3.13", help="interpreter for task envs (path or version)")
    ap.add_argument("--uv", default="uv")
    ap.add_argument("--wheels", type=Path, default=None, help="install task envs offline from this folder")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache/g4res_uv")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--deadline-min", type=float, default=None, help="start no new task after this many minutes")
    a = ap.parse_args()
    CFG.update(uv=a.uv, python=a.python, wheels=str(a.wheels.resolve()) if a.wheels else None)
    t_start = time.time()
    a.release = a.release.resolve()
    tasks = [json.loads(l) for l in (a.release / "tasks.jsonl").read_text().splitlines() if l.strip()]
    if a.tasks:
        tasks = [t for t in tasks if t["instance_id"] in set(a.tasks)]
    out_dir = a.out / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = a.cache
    cache.mkdir(parents=True, exist_ok=True)
    if a.model == "gold":
        def factory(task, ws):
            files = {}
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                checkout(a.release, task, d)
                (d / ".gold.patch").write_text(task["patch"])
                sh(["git", "apply", ".gold.patch"], cwd=d)
                _, names, _ = sh(["git", "status", "--porcelain"], cwd=d)
                for line in names.splitlines():
                    p = line[3:]
                    if p != ".gold.patch" and (d / p).is_file():
                        files[p] = (d / p).read_text()
            return GoldReplay(task, files)
    else:
        local = not a.endpoint.startswith("https://generativelanguage")
        key = "EMPTY" if local else a.key_file.expanduser().read_text().strip()
        extra = {}
        if a.max_tokens:
            extra["max_tokens"] = a.max_tokens
        if a.thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": True}
        shared = ChatModel(a.model, key, a.rpm, a.temperature, endpoint=a.endpoint, extra=extra)

        def factory(task, ws):
            return shared
    jobs = [(t, s) for s in range(a.samples) for t in tasks]

    def job(ts):
        if a.deadline_min and (time.time() - t_start) / 60 > a.deadline_min and \
                not (out_dir / f"{ts[0]['instance_id']}__s{ts[1]}.json").exists():
            return {"status": "skipped_deadline"}
        return solve(ts[0], ts[1], a, factory, out_dir, cache)

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(job, jobs))
    done = [r for r in results if r.get("status") == "done"]
    print(f"skipped (deadline): {sum(r.get('status') == 'skipped_deadline' for r in results)}")
    print(f"\n{a.run}: {sum(r['resolved'] for r in done)}/{len(done)} resolved "
          f"({len(results) - len(done)} errors; results in {out_dir})")


if __name__ == "__main__":
    main()
