"""Re-grade saved agent patches (run_gemma.py outputs) with the official swegemma Phase 2 rules.

Uses official_grade_check.grade() on each saved patch and writes <run dir>/<id>__s<k>.official.json.

usage: unshare -rn comp_data/venv_adk/bin/python -I scripts/official_regrade_runs.py --run-dir kaggle_out/core_final \
          --release release --python <py3.13> --uv <uv> --wheels release/wheels --workers 2
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_grade_check as og  # noqa: E402
import run_gemma as rg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--python", required=True)
    ap.add_argument("--uv", default="uv")
    ap.add_argument("--wheels", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/home/claude/uvcache_og"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    rg.CFG.update(uv=a.uv, python=a.python, wheels=str(a.wheels.resolve()))
    a.release = a.release.resolve()
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in (a.release / "tasks.jsonl").read_text().splitlines() if l.strip()}
    runs = sorted(p for p in a.run_dir.glob("*__s*.json") if not p.name.endswith(".official.json"))

    def one(p):
        out = p.with_name(p.stem + ".official.json")
        if out.exists():
            return json.loads(out.read_text())
        r = json.loads(p.read_text())
        patch = r.get("patch") or ""
        g = og.grade(tasks[r["instance_id"]], patch, a.release, a.cache, "agent") if patch.strip() else \
            {"resolved": False, "reason": "empty patch"}
        g.pop("output_tail", None)
        rec = {"instance_id": r["instance_id"], "sample": r.get("sample", 0), "ours": bool(r.get("resolved")), "official": g}
        out.write_text(json.dumps(rec, indent=1))
        print(f"{p.stem}: ours={rec['ours']} official={g.get('resolved')}", flush=True)
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, runs))
    from collections import Counter
    print(Counter((r["ours"], bool(r["official"].get("resolved"))) for r in res))


if __name__ == "__main__":
    main()
