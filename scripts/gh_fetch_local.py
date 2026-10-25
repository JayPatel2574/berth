"""Fetch PR + closing-issue data from GitHub on the user's own machine (stage 4, network part).

Reads the token from the file `.github_token` next to this repo's root (never printed, never written anywhere else),
reads `data/pr_queue.jsonl` ({"repo": "owner/name", "pr": 123} per line) and appends one result per PR to
`data/gh_results.jsonl`. Resumable: PRs already in the results file are skipped. Stops after --max-seconds so it fits
in a short shell call; run it again to continue.

One GraphQL call per PR returns the PR text and the issues GitHub records as closed by it (closingIssuesReferences).
When that list is empty, issues named with fixes/closes/resolves #N in the PR title or body are looked up (one call
each); plain #N mentions are recorded for the "issue_ref" tier.

usage: python3 scripts/gh_fetch_local.py [--max-seconds 160]
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = """query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){pullRequest(number:$n){
 title body createdAt mergedAt url author{login}
 closingIssuesReferences(first:5){nodes{number title body createdAt url author{login}}}}}}"""
QI = """query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){issueOrPullRequest(number:$n){
 __typename ... on Issue{number title body createdAt url author{login}}}}}"""
CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+(?:#|https://github\.com/[\w.-]+/[\w.-]+/issues/)(\d+)", re.I)
MENTION = re.compile(r"(?:^|[\s(])#(\d+)\b|https://github\.com/[\w.-]+/[\w.-]+/issues/(\d+)")


def gql(token, query, **variables):
    req = urllib.request.Request("https://api.github.com/graphql", method="POST",
                                 data=json.dumps({"query": query, "variables": variables}).encode(),
                                 headers={"Authorization": f"bearer {token}", "Content-Type": "application/json",
                                          "User-Agent": "g4res-fetch"})
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    if out.get("errors"):
        raise RuntimeError(str(out["errors"])[:300])
    return out["data"]["repository"]


def issue(token, o, r, n, kind):
    try:
        it = gql(token, QI, o=o, r=r, n=n)["issueOrPullRequest"]
    except RuntimeError:  # the number does not exist in this repo (e.g. it refers to another project)
        return None
    if not it or it.get("__typename") != "Issue":
        return None
    return {"kind": kind, "number": it["number"], "title": it["title"], "body": it["body"] or "",
            "created_at": it["createdAt"], "url": it["url"], "author": (it.get("author") or {}).get("login")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=160)
    a = ap.parse_args()
    token = (ROOT / ".github_token").read_text().strip()
    queue = [json.loads(l) for l in (ROOT / "data/pr_queue.jsonl").read_text().splitlines() if l.strip()]
    out_path = ROOT / "data/gh_results.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l)
                done.add((d["repo"], d["pr"]))
    t0, n_new, n_err = time.time(), 0, 0
    with out_path.open("a") as out:
        for q in queue:
            if (q["repo"], q["pr"]) in done:
                continue
            if time.time() - t0 > a.max_seconds:
                break
            o, r = q["repo"].split("/")
            try:
                p = gql(token, Q, o=o, r=r, n=q["pr"])["pullRequest"]
                issues = [{"kind": "issue", "number": i["number"], "title": i["title"], "body": i["body"] or "",
                           "created_at": i["createdAt"], "url": i["url"], "author": (i.get("author") or {}).get("login")}
                          for i in p["closingIssuesReferences"]["nodes"]]
                text = f"{p['title']}\n{p['body'] or ''}\n{q.get('commit_msg', '')}"
                if not issues:
                    for num in list(dict.fromkeys(int(x) for x in CLOSES.findall(text)))[:5]:
                        it = issue(token, o, r, num, "issue")
                        if it:
                            issues.append(it)
                mentions = [int(x or y) for x, y in MENTION.findall(text)]
                rec = {"repo": q["repo"], "pr": q["pr"], "title": p["title"], "body": p["body"] or "",
                       "created_at": p["createdAt"], "merged_at": p["mergedAt"], "url": p["url"],
                       "author": (p.get("author") or {}).get("login"), "issues": issues,
                       "mentions": sorted(set(mentions) - {q["pr"]} - {i["number"] for i in issues})}
                if not issues:  # issue_ref tier: plain mentions of an issue opened before the PR
                    for num in rec["mentions"][:3]:
                        it = issue(token, o, r, num, "issue_ref")
                        if it and it["created_at"] < rec["created_at"]:
                            issues.append(it)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_new += 1
            except Exception as e:  # noqa: BLE001  (never print the token; errors come from the API)
                n_err += 1
                print(f"{q['repo']}#{q['pr']}: {type(e).__name__}: {str(e)[:150]}")
    left = sum(1 for q in queue if (q["repo"], q["pr"]) not in done) - n_new
    print(f"fetched {n_new}, errors {n_err}, remaining {left}")


if __name__ == "__main__":
    main()
