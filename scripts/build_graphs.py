"""Build official-format graphs (and stand-in embeddings) for the base commits of verified tasks.

Same outputs as package_task.py's graph step, without the rest of the packaging, so localization baselines can run
early. Skips graphs that already exist.

usage: python -I build_graphs.py <verify dir> --release release [--sources issue issue_ref]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))
import code_graph  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify_dir", type=Path)
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--repos", type=Path, default=Path("repos"))
    ap.add_argument("--sources", nargs="*", default=None)
    a = ap.parse_args()
    (a.release / "graphs").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(a.verify_dir.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("status") != "ok" or (a.sources and rec.get("statement_source") not in a.sources):
            continue
        stem = f"{rec['repo'].split('/')[1]}_{rec['base_commit']}"
        out = a.release / "graphs" / f"{stem}.json"
        if out.exists():
            continue
        tmp = Path(tempfile.mkdtemp(prefix="g_"))
        try:
            arch = subprocess.Popen(["git", "-C", str(a.repos / rec["repo"].replace("/", "_")), "archive",
                                     rec["base_commit"]], stdout=subprocess.PIPE)
            subprocess.run(["tar", "-x", "-C", str(tmp)], stdin=arch.stdout, check=True)
            arch.wait()
            nodes, links, _ = code_graph.build(tmp, seed=stem)
            out.write_text(code_graph.to_json(code_graph.to_graph(stem, nodes, links)))
            n += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"built {n} graphs")


if __name__ == "__main__":
    main()
