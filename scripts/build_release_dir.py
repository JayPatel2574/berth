"""Assemble the public Berth dataset folder (Hugging Face / Kaggle) from the working release.

berth_dataset/
  README.md                       dataset card (written separately: paper/dataset_card.md)
  tasks.jsonl                     all 653 tasks (official tasks.jsonl columns + Berth fields)
  core.jsonl, extended.jsonl, overlap.jsonl
  snapshots/<instance_id>.tgz     (Kaggle variant: snapshots.tar, see --kaggle)
  graphs/<repo>_<commit>.json, embeddings/<repo>_<commit>.npz
  wheels/                         (Kaggle variant: wheels.tar)
  runs/gemma-4-31b-it-qat-w4a16-ct/<id>__s0.json, .traj.jsonl, .official.json
  results/localization_core.jsonl, results/paper_numbers.json
  LICENSES/                       source-repository licence list and the pipeline's licence

--kaggle packs snapshots/ and wheels/ into plain tar files, because Kaggle unpacks archives found inside datasets
(every snapshot unpacks to the same top-level paths).

usage: python -I build_release_dir.py --out /home/claude/berth_dataset [--kaggle]
"""
import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kaggle", action="store_true")
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    a.out.mkdir(parents=True)
    rel = ROOT / "release"
    rows = [json.loads(l) for l in (rel / "tasks.jsonl").read_text().splitlines() if l.strip()]
    (a.out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for sp in ("core", "extended", "overlap"):
        (a.out / f"{sp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == sp))
    for x in sorted({r["graph"] for r in rows} | {r["embeddings"] for r in rows}):
        link(rel / x, a.out / x)
    if a.kaggle:
        with tarfile.open(a.out / "snapshots.tar", "w") as t:
            for r in rows:
                t.add(rel / "snapshots" / f"{r['instance_id']}.tgz", arcname=f"snapshots/{r['instance_id']}.tgz")
        with tarfile.open(a.out / "wheels.tar", "w") as t:
            for w in sorted((rel / "wheels").iterdir()):
                t.add(w, arcname=f"wheels/{w.name}")
    else:
        for r in rows:
            link(rel / "snapshots" / f"{r['instance_id']}.tgz", a.out / "snapshots" / f"{r['instance_id']}.tgz")
        for w in sorted((rel / "wheels").iterdir()):
            link(w, a.out / "wheels" / w.name)
    run_dir = a.out / "runs" / "gemma-4-31b-it-qat-w4a16-ct"
    for f in sorted((ROOT / "kaggle_out/core_final").iterdir()):
        link(f, run_dir / f.name)
    link(ROOT / "data/loc_results_core.jsonl", a.out / "results/localization_core.jsonl")
    link(ROOT / "data/paper_numbers.json", a.out / "results/paper_numbers.json")
    lic = a.out / "LICENSES"
    lic.mkdir()
    (lic / "SOURCE_REPOSITORIES.md").write_text(
        "Each snapshot is a copy of the named repository at `base_commit` and keeps that repository's own licence "
        "file. All are permissive:\n\n" + "\n".join(
            f"- {repo}" for repo in sorted({r["repo"] for r in rows})) +
        "\n\nLicences found: MIT, BSD (2/3-clause), Apache-2.0; structlog is dual MIT/Apache-2.0.\n"
        "Berth's own annotations (task metadata, localization labels, flags, results) and pipeline code: Apache-2.0.\n"
        "Code graphs were built with code_graph.py from github.com/nakranivaibhav/gemma-4-developer-agent (Apache-2.0).\n")
    shutil.copy(ROOT / "scripts/third_party/LICENSE-nakranivaibhav", lic / "LICENSE-Apache-2.0.txt")
    card = ROOT / "paper/dataset_card.md"
    if card.exists():
        shutil.copy(card, a.out / "README.md")
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file())
    print(f"{a.out}: {sum(1 for f in a.out.rglob('*') if f.is_file())} files, {size / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
