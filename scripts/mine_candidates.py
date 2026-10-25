"""Stage 1: find candidate task commits in cloned repos from git history alone (no GitHub API).

A candidate is a first-parent commit on the default branch that
  - changes at least one test .py file and 1..MAX_SRC non-test source .py files,
  - references a PR number (squash "(#123)" or "Merge pull request #123"),
  - is not a pure docs/CI/typing change.
Output: one JSON line per candidate to stdout.

usage: python -I mine_candidates.py <repos_dir> --since 2023-01-01 > candidates.jsonl
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

MAX_SRC = 3
PR_RE = re.compile(r"\(#(\d+)\)|Merge pull request #(\d+)|\bPR #(\d+)")
SKIP_MSG = re.compile(r"^(docs?|ci|chore|build|style|typing|bump|release|pre-commit|\[pre-commit)", re.I)
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
NON_SRC = re.compile(r"(^|/)(docs?|docs_src|examples?|benchmarks?|scripts|tools|\.github|noxfile\.py|setup\.py|conftest\.py)")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos_dir", type=Path)
    ap.add_argument("--since", default="2023-01-01")
    a = ap.parse_args()
    for repo in sorted(p for p in a.repos_dir.iterdir() if (p / ".git").exists()):
        slug = git(repo, "remote", "get-url", "origin").strip().removeprefix("https://github.com/").removesuffix(".git")
        log = git(repo, "log", "--first-parent", f"--since={a.since}", "--format=%x00%H%x01%P%x01%cI%x01%s", "--name-only")
        for block in log.split("\x00")[1:]:
            head, _, files = block.partition("\n")
            sha, parents, date, subject = head.split("\x01", 3)
            files = [f for f in files.split("\n") if f.strip()]
            if SKIP_MSG.match(subject):
                continue
            m = PR_RE.search(subject)
            if not m:
                continue
            pr = int(next(g for g in m.groups() if g))
            py = [f for f in files if f.endswith(".py")]
            tests = [f for f in py if TEST_RE.search(f)]
            src = [f for f in py if f not in tests and not NON_SRC.search(f)]
            if not tests or not (1 <= len(src) <= MAX_SRC):
                continue
            print(json.dumps({
                "repo": slug, "sha": sha, "parent": parents.split()[0], "date": date, "pr": pr,
                "subject": subject, "src_files": src, "test_files": tests, "n_files": len(files),
            }))


if __name__ == "__main__":
    main()
