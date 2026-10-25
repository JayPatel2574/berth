"""Quality flags for verified tasks (stored in rec["quality"], exported per task).

new_api_in_tests      names the gold patch defines (localization.nodes_added, last component) that the added test
                      lines use: the hidden tests depend on an interface the fix invents
new_api_unstated      those names that the problem statement never mentions -> the task may be underspecified
                      (a correct fix with another name fails); `underspecified` = bool(new_api_unstated)
f2p_count, patch_lines, files_changed     size measures
statement_chars       length of the problem statement

usage: python -I quality_flags.py <verify dir>
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def added_lines(diff):
    return [l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_dir", type=Path)
    a = ap.parse_args()
    c = Counter()
    for f in sorted(a.verify_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") != "ok" or not rec.get("localization"):
            continue
        new = {n.rsplit(".", 1)[-1] for n in rec["localization"].get("nodes_added", [])}
        new = {n for n in new if not (n.startswith("__") and n.endswith("__")) and not n.startswith("test")}
        test_ids = set(IDENT.findall("\n".join(added_lines(rec.get("test_patch", "")))))
        used = sorted(new & test_ids)
        stmt = rec.get("problem_statement") or ""
        unstated = [n for n in used if not re.search(rf"\b{re.escape(n)}\b", stmt)]
        patch_add = added_lines(rec.get("patch", ""))
        rec["quality"] = {"new_api_in_tests": used, "new_api_unstated": unstated, "underspecified": bool(unstated),
                          "f2p_count": len(rec.get("FAIL_TO_PASS", [])), "patch_lines": len(patch_add),
                          "files_changed": len(rec["localization"].get("files", [])), "statement_chars": len(stmt)}
        f.write_text(json.dumps(rec, indent=1))
        c["tasks"] += 1
        c["underspecified"] += bool(unstated)
        c["new_api_in_tests"] += bool(used)
    print(dict(c))


if __name__ == "__main__":
    main()
