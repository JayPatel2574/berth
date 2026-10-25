"""Stability pass: fold repeated gold-replay runs (run_gemma.py --model gold --samples N, run in the target
environment: offline, portable Python 3.13, parallel workers) back into the task records.

  - a PASS_TO_PASS test that failed in any gold run -> moved to PASS_TO_PASS_flaky (not graded)
  - a FAIL_TO_PASS test that failed in any gold run, or a run that errored/did not resolve for F2P -> task unstable
Writes rec["stability"] = {"runs": n, "status": "ok" | "unstable", "flaky_p2p": [...], "f2p_failed": [...]}

usage: python -I stability.py <verify dir> <gold run dir> --samples 3
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--samples", type=int, default=3)
    a = ap.parse_args()
    seen = {}
    for f in a.run_dir.glob("*__s*.json"):
        r = json.loads(f.read_text())
        seen.setdefault(r["instance_id"], []).append(r)
    counts = {"ok": 0, "unstable": 0, "incomplete": 0}
    for iid, runs in sorted(seen.items()):
        vf = a.verify_dir / f"{iid}.json"
        rec = json.loads(vf.read_text())
        if len(runs) < a.samples:
            counts["incomplete"] += 1
            continue
        flaky = sorted({t for r in runs for t in r.get("p2p_failed", [])})
        f2p_bad = sorted({t for r in runs for t in r.get("f2p_failed", [])})
        broken = [r for r in runs if not r.get("applied", False) or r.get("f2p_passed") != r.get("f2p_total")]
        status = "unstable" if (f2p_bad or broken) else "ok"
        rec["stability"] = {"runs": len(runs), "status": status, "flaky_p2p": flaky, "f2p_failed": f2p_bad}
        vf.write_text(json.dumps(rec, indent=1))
        counts[status] += 1
    print(counts)


if __name__ == "__main__":
    main()
