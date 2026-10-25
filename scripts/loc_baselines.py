"""Localization baselines over the official-format graphs (CPU only).

For each task with localization labels and a problem statement, rank the base commit's graph nodes (non-test nodes;
graph = release/graphs/<repo>_<base>.json) against the problem statement and score against the gold labels.

Rankers
  bm25        BM25 over node id tokens + node text (identifiers split on _ . and camelCase)
  tfidf_name  TF-IDF over node ids only (the forum's 52%-top-10 baseline on the official tasks)
  bm25+graph  BM25 scores smoothed over the graph: personalized PageRank seeded with the BM25 top 20 (alpha 0.5),
              so the graph tools' neighbourhoods can lift a node the text alone misses
  random      uniform shuffle (seeded), as a floor

Metrics per task: node hit@k (any gold edited node in the top k, k = 1, 5, 10, 20), file hit@k (the file of any of
the top-k nodes is a gold file) and node recall@10. Gold = nodes_edited (base-graph nodes; added nodes cannot be
retrieved before they exist). Test code is excluded by file path (tests/, test_*.py, conftest.py).

usage: python -I loc_baselines.py --release release --splits core --out data/loc_results.jsonl
"""
import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

TEST_PATH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


def test_modules(repo_dir, commit):
    """Module names (official naming rule) of the test files at `commit`, decided by file PATH, so library modules
    that merely have test-like names (e.g. click.testing) stay rankable."""
    import subprocess
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from localize import module_name, source_roots
    paths = [p for p in subprocess.run(["git", "-C", str(repo_dir), "ls-tree", "-r", "--name-only", commit],
                                       capture_output=True, text=True, check=True).stdout.split("\n") if p.endswith(".py")]
    roots = source_roots(paths)
    return {module_name(p, roots) for p in paths if TEST_PATH.search(p)}
TOK = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
STOP = set("the a an and or of to in is it be for on with as that this by not are was if at from but we i you can "
           "when should would will has have def self return none true false import class".split())
KS = (1, 5, 10, 20)


def toks(s):
    return [t.lower() for t in TOK.findall(s or "") if t.lower() not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = [Counter(d) for d in docs]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, q):
        q = Counter(q)
        out = []
        for d, L in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


def tfidf_scores(docs, q):
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    idf = {t: math.log((n + 1) / (f + 1)) + 1 for t, f in df.items()}
    qv = Counter(t for t in q if t in idf)
    qn = math.sqrt(sum((c * idf[t]) ** 2 for t, c in qv.items())) or 1
    out = []
    for d in docs:
        dv = Counter(d)
        dn = math.sqrt(sum((c * idf[t]) ** 2 for t, c in dv.items())) or 1
        out.append(sum(qv[t] * idf[t] * dv[t] * idf[t] for t in qv if t in dv) / (qn * dn))
    return out


def ppr(adj, seeds, alpha=0.5, iters=20):
    """Personalized PageRank; seeds {node index: weight}."""
    tot = sum(seeds.values()) or 1
    p = {i: w / tot for i, w in seeds.items()}
    r = dict(p)
    for _ in range(iters):
        nxt = defaultdict(float)
        for i, v in r.items():
            nb = adj[i]
            if nb:
                share = (1 - alpha) * v / len(nb)
                for j in nb:
                    nxt[j] += share
        for i, w in p.items():
            nxt[i] += alpha * w
        r = nxt
    return r


def evaluate(order, ids, gold_nodes, gold_files, node_file):
    ranked = [ids[i] for i in order]
    m = {}
    for k in KS:
        top = ranked[:k]
        m[f"node@{k}"] = int(any(x in gold_nodes for x in top))
        m[f"file@{k}"] = int(any(node_file.get(x) in gold_files for x in top))
    m["recall@10"] = len(set(ranked[:10]) & gold_nodes) / len(gold_nodes)
    m["gold_rank"] = next((i + 1 for i, x in enumerate(ranked) if x in gold_nodes), None)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, default=Path("release"))
    ap.add_argument("--out", type=Path, default=Path("data/loc_results.jsonl"))
    ap.add_argument("--splits", nargs="*", default=["core"])
    ap.add_argument("--repos", type=Path, default=Path("repos"))
    a = ap.parse_args()
    rows = []
    tasks = [json.loads(l) for l in (a.release / "tasks.jsonl").read_text().splitlines() if l.strip()]
    for rec in tasks:
        if a.splits and rec["split"] not in a.splits:
            continue
        loc = rec.get("localization")
        if not loc or not rec.get("problem_statement"):
            continue
        g_path = a.release / "graphs" / f"{rec['repo'].split('/')[1]}_{rec['base_commit']}.json"
        if not g_path.exists():
            continue
        g = json.loads(g_path.read_text())
        tmods = test_modules(a.repos / rec["repo"].replace("/", "_"), rec["base_commit"])
        is_test = lambda x: any(x == m or x.startswith(m + ".") for m in tmods)
        ids = [n["id"] for n in g["nodes"] if not is_test(n["id"])]
        index = {x: i for i, x in enumerate(ids)}
        text = {n["id"]: n.get("text") or "" for n in g["nodes"]}
        gold = set(loc["nodes_edited"]) & set(index)
        if not gold:
            continue
        # node -> module file, by matching module prefixes of gold files (only needed for file-level metrics)
        mods = {}
        for p in loc["files"]:
            parts = p[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            for i in range(len(parts)):
                mods[".".join(parts[i:])] = p
        node_file = {}
        for x in ids:
            for m, p in mods.items():
                if x.startswith(m + "."):
                    node_file[x] = p
                    break
        gold_files = set(loc["files"])
        q = toks(rec["problem_statement"])
        name_docs = [toks(x.replace(".", " ")) for x in ids]
        full_docs = [nd + toks(text[x]) for nd, x in zip(name_docs, ids)]
        adj = defaultdict(set)
        for e in g.get("edges", g.get("links", [])):
            s, t = index.get(e["source"]), index.get(e["target"])
            if s is not None and t is not None and s != t:
                adj[s].add(t)
        res = {"instance_id": rec["instance_id"], "repo": rec["repo"], "statement_source": rec.get("statement_source"), "split": rec["split"],
               "n_nodes": len(ids), "n_gold": len(gold)}
        bm = BM25(full_docs).scores(q)
        orders = {
            "bm25": sorted(range(len(ids)), key=lambda i: -bm[i]),
            "tfidf_name": (lambda s: sorted(range(len(ids)), key=lambda i: -s[i]))(tfidf_scores(name_docs, q)),
        }
        seeds = {i: bm[i] for i in orders["bm25"][:20] if bm[i] > 0}
        pr = ppr(adj, seeds) if seeds else {}
        orders["bm25+graph"] = sorted(range(len(ids)), key=lambda i: (-pr.get(i, 0.0), -bm[i]))
        rnd = list(range(len(ids)))
        random.Random(rec["instance_id"]).shuffle(rnd)
        orders["random"] = rnd
        for name, order in orders.items():
            res[name] = evaluate(order, ids, gold, gold_files, node_file)
        rows.append(res)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # summary
    print(f"{len(rows)} tasks")
    for name in ("random", "tfidf_name", "bm25", "bm25+graph"):
        ms = [r[name] for r in rows]
        if ms:
            print(f"{name:11s} " + "  ".join(f"{k}={sum(m[k] for m in ms) / len(ms):.2f}" for k in
                                            ("node@1", "node@5", "node@10", "node@20", "file@1", "file@5", "recall@10")))


if __name__ == "__main__":
    main()
