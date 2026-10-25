"""Every number quoted in the Berth paper, recomputed from the data files (one JSON, data/paper_numbers.json).

Inputs: data/candidates.jsonl, data/queue_recent.txt, data/verify/*.json, data/overlap_swerebench.json,
release/tasks.jsonl, release/graphs/, data/loc_results_core.jsonl, kaggle_out/core_final/*__s0.json,
runs/stability/*__s*.json.

usage: python -I paper_numbers.py
"""
import collections
import glob
import json
import random
import statistics
from pathlib import Path

D = Path("data")
rng = random.Random(0)


def ci(v, B=4000):
    rng = random.Random(0)  # seeded per call: each interval is reproducible regardless of call order
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"mean": round(sum(v) / n, 4), "lo": round(bs[int(.025 * B)], 4), "hi": round(bs[int(.975 * B)], 4), "n": n}


def main():
    out = {}
    cands = [json.loads(l) for l in open(D / "candidates.jsonl")]
    queue = [int(x) for x in open(D / "queue_recent.txt").read().split()]
    out["repos_cloned"] = len({c["repo"] for c in cands} | set(open("scripts/repos.txt").read().split()))
    out["candidates_since_2023"] = len(cands)
    out["candidates_since_2025_07"] = len(queue)
    recs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(D / "verify/*.json")))]
    out["verified_records"] = len(recs)
    out["verify_status"] = dict(collections.Counter(r["status"] for r in recs))
    ok = [r for r in recs if r["status"] == "ok"]
    out["statement_source_of_ok"] = dict(collections.Counter(r.get("statement_source", "missing") for r in ok))
    out["offline_status_of_ok_with_statement"] = dict(collections.Counter(
        (r.get("statement_source"), r.get("offline", {}).get("status")) for r in ok if r.get("statement_source")).most_common())
    out["offline_status_of_ok_with_statement"] = {f"{a}|{b}": c for (a, b), c in out["offline_status_of_ok_with_statement"].items()}
    out["stability_status"] = dict(collections.Counter(r.get("stability", {}).get("status") for r in ok if "stability" in r))
    out["tasks_with_flaky_p2p_removed"] = sum(bool(r.get("stability", {}).get("flaky_p2p")) for r in ok)
    stab = [json.loads(Path(f).read_text()) for f in glob.glob("runs/stability/*__s*.json")]
    out["gold_replays"] = {"runs": len(stab), "resolved": sum(bool(s.get("resolved")) for s in stab)}
    out["overlap_swerebench"] = len(json.loads((D / "overlap_swerebench.json").read_text()))

    rows = [json.loads(l) for l in open("release/tasks.jsonl")]
    out["release_split"] = dict(collections.Counter(r["split"] for r in rows))
    out["release_repos"] = len({r["repo"] for r in rows})
    core = [r for r in rows if r["split"] == "core"]
    ext = [r for r in rows if r["split"] == "extended"]
    out["core_repos"] = len({r["repo"] for r in core})
    out["core_per_repo"] = dict(collections.Counter(r["repo"] for r in core).most_common())
    out["core_statement_source"] = dict(collections.Counter(r["statement_source"] for r in core))
    out["post_cutoff"] = {"core": sum(r["post_cutoff"] for r in core), "extended": sum(r["post_cutoff"] for r in ext)}
    commit_date = {r["instance_id"]: r["created_at"] for r in recs}  # fix-commit date on the default branch
    cd = [commit_date[r["instance_id"]] for r in core]
    out["core_fix_commit_range"] = [min(cd)[:10], max(cd)[:10]]
    out["core_pr_not_marked_merged"] = sum(not r["merged_at"] for r in core)
    out["core_issue_created_range"] = [min(r["issue_created_at"] for r in core if r.get("issue_created_at"))[:10],
                                       max(r["issue_created_at"] for r in core if r.get("issue_created_at"))[:10]]
    med = lambda xs: statistics.median(xs)
    q = lambda r: r["quality"]
    out["core_medians"] = {
        "patch_added_lines": med([q(r)["patch_lines"] for r in core]),
        "files_changed": med([q(r)["files_changed"] for r in core]),
        "fail_to_pass": med([len(r["FAIL_TO_PASS"]) for r in core]),
        "pass_to_pass": med([len(r["PASS_TO_PASS"]) for r in core]),
        "statement_chars": med([q(r)["statement_chars"] for r in core]),
        "nodes_edited": med([len(r["localization"]["nodes_edited"]) for r in core]),
    }
    out["core_files_changed"] = dict(sorted(collections.Counter(q(r)["files_changed"] for r in core).items()))
    out["core_underspecified"] = sum(q(r)["underspecified"] for r in core)
    out["core_with_added_nodes"] = sum(bool(r["localization"]["nodes_added"]) for r in core)
    out["core_module_level_only"] = sum(not r["localization"]["nodes_edited"] for r in core)
    gsizes = []
    for g in sorted({r["graph"] for r in core}):
        d = json.loads(Path("release", g).read_text())
        gsizes.append(len(d["nodes"]))
    out["core_graph_nodes"] = {"graphs": len(gsizes), "median": med(gsizes), "min": min(gsizes), "max": max(gsizes)}
    out["release_bytes"] = {p: sum(f.stat().st_size for f in Path("release", p).rglob("*") if f.is_file())
                            for p in ("snapshots", "graphs", "embeddings", "wheels")}

    loc = [json.loads(l) for l in open(D / "loc_results_core.jsonl")]
    out["loc_tasks"] = len(loc)
    out["loc"] = {m: {k: ci([r[m][k] for r in loc]) for k in ("node@1", "node@5", "node@10", "node@20", "file@1", "file@5", "recall@10")}
                  for m in ("random", "tfidf_name", "bm25", "bm25+graph")}
    out["loc_graph_minus_bm25"] = {k: ci([r["bm25+graph"][k] - r["bm25"][k] for r in loc]) for k in ("node@1", "node@10", "file@1", "file@5")}

    tasks = {r["instance_id"]: r for r in rows}
    res = [json.loads(Path(f).read_text()) for f in glob.glob("kaggle_out/core_final/*__s0.json")]
    T = lambda r: tasks[r["instance_id"]]
    R = lambda rs: ci([int(bool(r["resolved"])) for r in rs])
    out["gemma"] = {
        "graded": len(res),
        "all": R(res),
        "post_cutoff": R([r for r in res if T(r)["post_cutoff"]]),
        "pre_cutoff": R([r for r in res if not T(r)["post_cutoff"]]),
        "underspecified": R([r for r in res if q(T(r))["underspecified"]]),
        "not_underspecified": R([r for r in res if not q(T(r))["underspecified"]]),
        "files_1": R([r for r in res if q(T(r))["files_changed"] == 1]),
        "files_2": R([r for r in res if q(T(r))["files_changed"] == 2]),
        "files_3": R([r for r in res if q(T(r))["files_changed"] == 3]),
        "post_cutoff_not_underspecified": R([r for r in res if T(r)["post_cutoff"] and not q(T(r))["underspecified"]]),
        "submitted": sum(bool(r.get("submitted")) for r in res),
        "hit_time_limit": sum(r["seconds"] >= 1200 for r in res),
        "hit_turn_limit": sum(r.get("turns") == 40 for r in res),
        "median_turns": med([r.get("turns", 0) for r in res]),
        "median_seconds": med([r["seconds"] for r in res]),
        "median_completion_tokens": med([r["usage"]["completion_tokens"] for r in res]),
        "empty_patch": sum(bool(r.get("empty")) for r in res),
        "resolved_by_repo": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(
            collections.Counter().items())},
    }
    by = collections.defaultdict(list)
    for r in res:
        by[T(r)["repo"]].append(int(bool(r["resolved"])))
    out["gemma"]["resolved_by_repo"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items(), key=lambda x: -len(x[1]))}
    # does the model fix the right code? overlap of edited files with gold files, resolved vs not
    def files_of(patch):
        return {l[6:] for l in patch.split("\n") if l.startswith("+++ b/")}
    hit = [int(bool(files_of(r.get("patch", "")) & set(T(r)["localization"]["files"]))) for r in res if r.get("patch")]
    out["gemma"]["edited_a_gold_file"] = ci(hit)
    out["gemma"]["unresolved_but_edited_gold_file"] = ci([int(bool(files_of(r.get("patch", "")) & set(T(r)["localization"]["files"])))
                                                          for r in res if r.get("patch") and not r["resolved"]])
    # ---- official swegemma Phase 2 rules ----
    og = {json.loads(Path(f).read_text())["instance_id"]: json.loads(Path(f).read_text())
          for f in glob.glob(str(D / "official_grade/*.json"))}
    out["official_checked"] = dict(collections.Counter(r["split"] for r in rows if r["instance_id"] in og))
    out["official_compatible"] = dict(collections.Counter(
        f"{r['split']}|{r.get('harness_compatible')}" for r in rows))
    async_ids = set(json.loads((D / "official_incompatible_core.json").read_text())["async"])
    inc = [r for r in core if r.get("harness_compatible") is False]
    out["core_incompatible"] = {"total": len(inc), "async_tests": sum(r["instance_id"] in async_ids for r in inc),
                                "other": sum(r["instance_id"] not in async_ids for r in inc),
                                "ids_other": sorted(r["instance_id"] for r in inc if r["instance_id"] not in async_ids)}
    out["env_pinned"] = {"tasks": sum(r.get("env_pinned_to_commit_date", False) for r in rows),
                         "compatible": sum(r.get("env_pinned_to_commit_date", False) and bool(r.get("harness_compatible")) for r in rows)}
    out["validator_pruned"] = {"tasks": sum(bool(r["FAIL_TO_PASS_dropped_validator"] or r["PASS_TO_PASS_dropped_validator"]) for r in rows),
                               "dropped_f2p": sum(len(r["FAIL_TO_PASS_dropped_validator"]) for r in rows),
                               "dropped_p2p": sum(len(r["PASS_TO_PASS_dropped_validator"]) for r in rows)}
    compat = [r for r in core if r.get("harness_compatible")]
    out["core_compatible"] = {"n": len(compat), "post_cutoff": sum(r["post_cutoff"] for r in compat),
                              "repos": len({r["repo"] for r in compat}),
                              "underspecified": sum(r["quality"]["underspecified"] for r in compat)}
    offi = {}
    for f in glob.glob("kaggle_out/core_final/*__s0.official.json"):
        o = json.loads(Path(f).read_text())
        offi[o["instance_id"]] = o
    if offi:
        cid = {r["instance_id"] for r in compat}
        O = lambda pred: ci([int(bool(offi[i]["official"].get("resolved"))) for i in sorted(offi) if i in cid and pred(tasks[i])])
        out["gemma_official"] = {
            "regraded": len(offi),
            "compatible_all": O(lambda t: True),
            "compatible_post_cutoff": O(lambda t: t["post_cutoff"]),
            "compatible_pre_cutoff": O(lambda t: not t["post_cutoff"]),
            "compatible_underspecified": O(lambda t: t["quality"]["underspecified"]),
            "compatible_not_underspecified": O(lambda t: not t["quality"]["underspecified"]),
            "compatible_files_1": O(lambda t: t["quality"]["files_changed"] == 1),
            "compatible_files_2": O(lambda t: t["quality"]["files_changed"] == 2),
            "compatible_files_3": O(lambda t: t["quality"]["files_changed"] == 3),
            "agreement_ours_vs_official_on_compatible": dict(collections.Counter(
                f"ours={o['ours']}|official={bool(o['official'].get('resolved'))}" for i, o in offi.items() if i in cid)),
        }
        # localization failure share on compatible tasks, official grading
        unres = [r for r in res if r["instance_id"] in cid and r.get("patch") and not offi[r["instance_id"]]["official"].get("resolved")]
        out["gemma_official"]["unresolved_no_gold_file"] = ci([int(not (files_of(r["patch"]) & set(T(r)["localization"]["files"]))) for r in unres])
    (D / "paper_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
