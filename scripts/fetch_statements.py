"""Stage 4: problem statements. Two steps, because GitHub API access runs on the user's machine:

  python -I fetch_statements.py queue <verify dir>   -> data/pr_queue.jsonl (PRs of ok tasks not fetched yet)
  (on the user's machine)  python3 scripts/gh_fetch_local.py   -> data/gh_results.jsonl
  python -I fetch_statements.py merge <verify dir>   -> problem_statement etc. into the task records

The problem statement follows SWE-bench: the resolved issue's title and body. statement_source is
  "issue"     GitHub's closing-issue link, or a fixes/closes/resolves #N keyword in the PR title/body/commit message
  "issue_ref" no closing link; a plain #N mention of an issue opened before the PR
  "pr"        no issue; the PR title/body (often describes the fix, so a lower tier, excluded from the core split)
"""
import argparse
import json
import subprocess
from pathlib import Path

QUEUE, RESULTS = Path("data/pr_queue.jsonl"), Path("data/gh_results.jsonl")


def load_results():
    out = {}
    if RESULTS.exists():
        for l in RESULTS.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                out[(d["repo"], d["pr"])] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["queue", "merge"])
    ap.add_argument("verify_dir", type=Path)
    a = ap.parse_args()
    res = load_results()
    recs = [(f, json.loads(f.read_text())) for f in sorted(a.verify_dir.glob("*.json"))]
    recs = [(f, r) for f, r in recs if r.get("status") == "ok"]
    if a.mode == "queue":
        q = []
        for _, r in recs:
            if (r["repo"], r["pr"]) in res:
                continue
            msg = subprocess.run(["git", "-C", f"repos/{r['repo'].replace('/', '_')}", "log", "-1", "--format=%B",
                                  r["fix_commit"]], capture_output=True, text=True).stdout
            q.append({"repo": r["repo"], "pr": r["pr"], "commit_msg": msg})
        QUEUE.write_text("".join(json.dumps(x) + "\n" for x in q))
        print(f"queued {len(q)}")
        return
    counts = {"issue": 0, "issue_ref": 0, "pr": 0, "missing": 0}
    for f, r in recs:
        d = res.get((r["repo"], r["pr"]))
        if not d:
            counts["missing"] += 1
            continue
        issues = [i for i in d["issues"] if i["created_at"] <= d["created_at"]] or \
                 [i for i in d["issues"] if i["kind"] == "issue"]  # a closing link to an issue filed later still counts
        if issues:
            r["problem_statement"] = "\n\n".join(f"{i['title']}\n{i['body']}".strip() for i in issues)
            r["statement_source"] = issues[0]["kind"]
            r["issue_urls"] = [i["url"] for i in issues]
            r["issue_created_at"] = min(i["created_at"] for i in issues)
        else:
            r["problem_statement"] = f"{d['title']}\n{d['body']}".strip()
            r["statement_source"] = "pr"
            for k in ("issue_urls", "issue_created_at"):
                r.pop(k, None)
        r.update(pr_url=d["url"], merged_at=d["merged_at"], pr_author=d["author"], pr_title=d["title"],
                 pr_body=d["body"], pr_created_at=d["created_at"])
        counts[r["statement_source"]] += 1
        f.write_text(json.dumps(r, indent=1))
    print(counts)


if __name__ == "__main__":
    main()
