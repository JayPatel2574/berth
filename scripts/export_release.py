"""Write release/tasks.jsonl from verified + packaged task records.

A task is released when it verified online (status ok) and passed the offline check (offline.status ok); with
--require-stability also the repeated gold-replay stability pass (stability.py). Flaky PASS_TO_PASS tests found by
that pass are moved to PASS_TO_PASS_flaky and not graded. harness_compatible comes from the official-rules check
(scripts/official_grade_check.py); its validator-pruned test lists replace FAIL_TO_PASS / PASS_TO_PASS.
Columns follow the official tasks.jsonl (instance_id, repo, base_commit, patch, test_patch, problem_statement,
hints_text, created_at) plus FAIL_TO_PASS / PASS_TO_PASS (offline-stable) and our extras.

Splits
  core      problem statement is a real issue (statement_source issue or issue_ref), not in any SWE-rebench release
  extended  PR-description statements (statement_source pr): fine for training, weaker as an evaluation
  (tasks that appear in SWE-rebench are kept with split "overlap")

usage: python -I export_release.py <verify dir> --release release
"""
import argparse
import json
from collections import Counter
from pathlib import Path

CUTOFF = "2025-02-01"  # Gemma 4 model card: pretraining data cutoff January 2025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_dir", type=Path)
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--require-stability", action="store_true", help="only tasks that passed the stability pass")
    a = ap.parse_args()
    overlap = {(o["repo"], o["pr"]) for o in json.loads(Path("data/overlap_swerebench.json").read_text())}
    rows = []
    for f in sorted(a.verify_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") != "ok" or r.get("offline", {}).get("status") != "ok" or "problem_statement" not in r:
            continue
        if a.require_stability and r.get("stability", {}).get("status") != "ok":
            continue
        flaky = set(r.get("stability", {}).get("flaky_p2p", []))
        og_path = Path("data/official_grade") / f"{r['instance_id']}.json"
        og = json.loads(og_path.read_text()) if og_path.exists() else None
        pruned = ((og or {}).get("gold") or {}).get("pruned") or {}
        f2p = pruned.get("FAIL_TO_PASS", r["FAIL_TO_PASS"])
        p2p = [t for t in pruned.get("PASS_TO_PASS", r.get("PASS_TO_PASS_offline", r["PASS_TO_PASS"])) if t not in flaky]
        src = r["statement_source"]
        split = "overlap" if (r["repo"].lower(), r["pr"]) in overlap else "core" if src in ("issue", "issue_ref") else "extended"
        rows.append({
            "instance_id": r["instance_id"], "repo": r["repo"], "base_commit": r["base_commit"],
            "patch": r["patch"], "test_patch": r["test_patch"], "problem_statement": r["problem_statement"],
            "hints_text": "", "created_at": r.get("pr_created_at") or r["created_at"],
            "FAIL_TO_PASS": f2p,
            "PASS_TO_PASS": p2p,
            "PASS_TO_PASS_flaky": sorted(flaky),
            # ids removed because the official validator's prefix matching collides with a skipped/failed sibling test
            "FAIL_TO_PASS_dropped_validator": pruned.get("dropped_f2p", []),
            "PASS_TO_PASS_dropped_validator": pruned.get("dropped_p2p", []),
            # official swegemma Phase 2 rules (scripts/official_grade_check.py): gold resolves AND empty patch does not
            "harness_compatible": None if og is None else bool(og["compatible"]),
            "env_pinned_to_commit_date": bool(r.get("env_pinned_to_commit_date")),
            # extras
            "split": split, "statement_source": src, "issue_urls": r.get("issue_urls", []), "pr_url": r.get("pr_url"),
            "fix_commit": r["fix_commit"], "merged_at": r.get("merged_at"),
            "graph": f"graphs/{r['repo'].split('/')[1]}_{r['base_commit']}.json",
            "embeddings": f"embeddings/{r['repo'].split('/')[1]}_{r['base_commit']}.npz",
            "snapshot": f"snapshots/{r['instance_id']}.tgz",
            "test_patch_files": r["test_patch_files"], "patch_files": r["patch_files"],
            "requirements": r["offline"]["requirements"], "build_requires": r["offline"].get("build_requires", []),
            "localization": r.get("localization"),
            "quality": r.get("quality"),
            "issue_created_at": r.get("issue_created_at"),
            # True when neither the issue nor the PR existed before Gemma 4's pretraining cutoff (January 2025)
            "post_cutoff": min(x for x in (r.get("issue_created_at"), r.get("pr_created_at") or r["created_at"]) if x)
                           >= CUTOFF,
        })
    a.release.mkdir(parents=True, exist_ok=True)
    (a.release / "tasks.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(len(rows), Counter(x["split"] for x in rows), Counter((x["split"], x["harness_compatible"]) for x in rows))
    print(Counter((x["split"], x["repo"]) for x in rows if x["split"] == "core").most_common(30))


if __name__ == "__main__":
    main()
