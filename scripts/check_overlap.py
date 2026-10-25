"""Decontamination: which candidate PRs already appear in public SWE task sets.

Downloads the parquet shards of nebius/SWE-rebench (filtered, test), nebius/SWE-rebench-V2 and nebius/SWE-rebench-V2-PRs
from the Hugging Face hub, matches (repo, PR number) against our candidates created on/after --since, and writes
data/overlap_swerebench.json. Overlapping tasks are kept out of the core split.

usage: python -I check_overlap.py --since 2025-07-01
"""
import argparse
import json
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

SETS = ["nebius/SWE-rebench", "nebius/SWE-rebench-V2", "nebius/SWE-rebench-V2-PRs"]


def shards(ds):
    out = subprocess.run(["curl", "-sSL", f"https://huggingface.co/api/datasets/{ds}/parquet"], capture_output=True,
                         text=True, check=True).stdout
    return [u for split in json.loads(out)["default"].values() for u in split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-07-01")
    ap.add_argument("--cache", type=Path, default=Path("data/ext"))
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open("data/candidates.jsonl")]
    recent = {(r["repo"].lower(), r["pr"]) for r in rows if r["date"] >= a.since}
    hits = {}
    for ds in SETS:
        for url in shards(ds):
            f = a.cache / (ds.replace("/", "_") + "_" + "_".join(url.split("/")[-3:]))
            if not f.exists():
                subprocess.run(["curl", "-sSL", "-o", str(f), url], check=True)
            names = pq.ParquetFile(f).schema_arrow.names
            cols = [c for c in ("instance_id", "repo", "pull_number") if c in names]
            for r in pq.read_table(f, columns=cols).to_pylist():
                num = r.get("pull_number")
                if num is None:
                    tail = r["instance_id"].rsplit("-", 1)[-1]
                    num = int(tail) if tail.isdigit() else None
                k = ((r.get("repo") or "").lower(), num)
                if k in recent:
                    hits.setdefault(k, set()).add(ds)
    out = [{"repo": k[0], "pr": k[1], "sources": sorted(v)} for k, v in sorted(hits.items())]
    Path("data/overlap_swerebench.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} of {len(recent)} candidates since {a.since} appear in {', '.join(SETS)}")


if __name__ == "__main__":
    main()
