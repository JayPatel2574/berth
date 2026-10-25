"""Stage 3: gold localization labels in official graph node ids.

For a verified task, map every changed line of the gold code patch (non-test .py files) to the innermost graph
node (def or class) that contains it, using the same node naming as the official graphs (scripts/third_party/
code_graph.py: module from the deepest source root, a single "current class", async defs are not nodes).

Labels written into the task record under "localization":
  files          changed non-test .py files
  nodes_edited   base-graph node ids whose source span contains a deleted/modified line, or the insertion point
                 of added lines
  nodes_added    node ids that exist only after the fix (new defs/classes)
  module_level   files where some change falls outside every node (imports, constants, async defs)
  aux_files      changed .py files outside the library (benchmarks, docs, scripts): not localization targets

usage: python -I localize.py <verify dir> --repos repos     (updates ok records in place)
"""
import argparse
import ast
import json
import re
import subprocess
from pathlib import PurePosixPath
from pathlib import Path

TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
NON_SRC = re.compile(r"(^|/)(docs?|docs_src|examples?|benchmarks?|scripts|tools|\.github|noxfile\.py|setup\.py)(/|$)")
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def source_roots(paths):
    """Same rule as code_graph.source_roots, from a path list: the repo root plus the parent of every top-level
    package (a folder with __init__.py whose parent folder has none)."""
    inits = {PurePosixPath(p).parent.parts for p in paths if p.endswith("__init__.py")}
    roots = {()}
    for d in inits:
        if d and d[:-1] not in inits:
            roots.add(d[:-1])
    return roots


def module_name(path, roots):
    p = PurePosixPath(path)
    dirs = p.parent.parts
    base = max((r for r in roots if dirs[:len(r)] == r), key=len)
    parts = list(p.with_suffix("").parts)[len(base):]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def node_spans(src, mod):
    """[(node_id, start, end)] for every def/class node, using the official naming quirks."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    spans = []

    class V(ast.NodeVisitor):
        cur = None

        def visit_ClassDef(self, n):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            spans.append((f"{mod}.{n.name}", start, n.end_lineno))
            self.cur = n.name
            self.generic_visit(n)
            self.cur = None

        def visit_FunctionDef(self, n):
            q = f"{mod}.{self.cur}.{n.name}" if self.cur else f"{mod}.{n.name}"
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            spans.append((q, start, n.end_lineno))
            self.generic_visit(n)

    V().visit(tree)
    return spans


def innermost(spans, line):
    best = None
    for q, s, e in spans:
        if s <= line <= e and (best is None or e - s < best[2] - best[1]):
            best = (q, s, e)
    return best[0] if best else None


def changed_lines(diff_text):
    """{path: (old_lines_touched, new_lines_added)} for a unified diff; for a pure insertion the old-side anchor is the
    line just before it."""
    out, path, old, new = {}, None, 0, 0
    for line in diff_text.split("\n"):
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            if path is not None:
                out.setdefault(path, (set(), set()))
            continue
        if line.startswith("--- "):
            continue
        m = HUNK.match(line)
        if m:
            old, new = int(m.group(1)), int(m.group(3))
            continue
        if path is None or not line:
            if path is not None and line == "":
                old += 1
                new += 1
            continue
        tag = line[0]
        if tag == "-":
            out[path][0].add(old)
            old += 1
        elif tag == "+":
            out[path][1].add(new)
            out[path][0].add(max(old - 1, 1))  # insertion anchor on the old side
            new += 1
        elif tag == " ":
            old += 1
            new += 1
    return out


def label(repo, rec):
    base, fix = rec["base_commit"], rec["fix_commit"]
    base_paths = git(repo, "ls-tree", "-r", "--name-only", base).split("\n")
    fix_paths = git(repo, "ls-tree", "-r", "--name-only", fix).split("\n")
    roots_b, roots_f = source_roots([p for p in base_paths if p.endswith(".py")]), source_roots([p for p in fix_paths if p.endswith(".py")])
    diff = git(repo, "diff", "-U0", base, fix, "--", *[f for f in rec["patch_files"] if f.endswith(".py") and not TEST_RE.search(f)]) \
        if any(f.endswith(".py") for f in rec["patch_files"]) else ""
    lines = changed_lines(diff)
    edited, added, module_level, files = [], [], [], []
    aux = sorted(p for p in lines if NON_SRC.search(p))
    for path, (old_l, new_l) in lines.items():
        if path in aux:
            continue
        files.append(path)
        old_src = git(repo, "show", f"{base}:{path}") if path in base_paths else ""
        new_src = git(repo, "show", f"{fix}:{path}") if path in fix_paths else ""
        old_spans = node_spans(old_src, module_name(path, roots_b)) or []
        new_spans = node_spans(new_src, module_name(path, roots_f)) or []
        old_ids = {q for q, _, _ in old_spans}
        outside = False
        # deletions/modifications and insertion anchors on the old side
        for ln in sorted(old_l):
            q = innermost(old_spans, ln) if old_src else None
            if q:
                edited.append(q)
            elif old_src:
                outside = True
        # added lines on the new side: inside a node that is new -> added; else attribute to that (existing) node
        for ln in sorted(new_l):
            q = innermost(new_spans, ln)
            if q is None:
                outside = True
            elif q in old_ids:
                edited.append(q)
            else:
                added.append(q)
        if outside:
            module_level.append(path)
    uniq = lambda xs: sorted(dict.fromkeys(xs))
    return {"files": sorted(files), "nodes_edited": uniq(edited), "nodes_added": uniq(added),
            "module_level": sorted(module_level), "aux_files": aux}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_dir", type=Path)
    ap.add_argument("--repos", type=Path, default=Path("repos"))
    a = ap.parse_args()
    n = 0
    for f in sorted(a.verify_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") != "ok" or "localization" in rec:
            continue
        repo = a.repos / rec["repo"].replace("/", "_")
        rec["localization"] = label(repo, rec)
        f.write_text(json.dumps(rec, indent=1))
        n += 1
    print(f"labelled {n}")


if __name__ == "__main__":
    main()
