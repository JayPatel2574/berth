"""Assemble the folder uploaded as a private Kaggle Dataset for the Gemma run (scripts/build_kaggle_notebook.py).

kaggle_bundle/
  g4res_bundle.json      marker + metadata (the notebook finds the dataset by this file)
  tasks.jsonl            released tasks of the chosen splits (needs --require-stability tasks by default)
  snapshots/<id>.snap    their repo snapshots (gzipped tar; opaque extension so Kaggle does not unpack them)
  tools/wheels.pyarch    the release's whole wheel pool (plain tar of wheels/)
  tools/                 portable CPython 3.13 (python-build-standalone, tools/python-3.13.pyarch = .tar.gz) + uv
  run_gemma.py           the agent runner

usage: python -I build_kaggle_bundle.py --release release --out kaggle_bundle --splits core
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--out", type=Path, default=Path("kaggle_bundle"))
    ap.add_argument("--splits", nargs="+", default=["core"])
    ap.add_argument("--tools", type=Path, default=Path("kaggle_tools"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in (a.release / "tasks.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] in a.splits]
    shutil.rmtree(a.out, ignore_errors=True)
    (a.out / "snapshots").mkdir(parents=True)
    (a.out / "wheels").mkdir()
    (a.out / "tools").mkdir()
    (a.out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        # opaque extension: Kaggle auto-extracts archives inside datasets, and every snapshot unpacks to the same paths
        shutil.copy(a.release / "snapshots" / f"{r['instance_id']}.tgz", a.out / "snapshots" / f"{r['instance_id']}.snap")
    # the whole wheel pool (~200 MB): a subset risks missing a build backend's own dependency
    # one opaque tar (wheels are zip files; Kaggle could unpack those too). The notebook extracts it to /tmp/wheels.
    keep = sorted((a.release / "wheels").iterdir())
    import tarfile
    with tarfile.open(a.out / "tools" / "wheels.pyarch", "w") as t:
        for w in keep:
            t.add(w, arcname=f"wheels/{w.name}")
    (a.out / "wheels").rmdir()
    shutil.copy(a.tools / "python-3.13.tar.gz", a.out / "tools" / "python-3.13.pyarch")
    shutil.copy(a.tools / "uv", a.out / "tools")
    shutil.copy(Path(__file__).resolve().parent / "run_gemma.py", a.out)
    meta = {"tasks": len(rows), "splits": a.splits, "wheels": len(keep), "built": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    (a.out / "g4res_bundle.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file()) / 2**20
    print(json.dumps(meta), f"{size:.0f} MB")


if __name__ == "__main__":
    main()
