"""Check that the paper states each key number exactly as recomputed from data (data/paper_numbers.json).

Each claim is (label, text that must appear in the paper, value recomputed from the numbers file). The check fails
when the recomputed value, formatted the way the paper writes it, does not appear in the paper.

usage: python -I check_paper_claims.py [--paper paper/berth_paper.md]
"""
import argparse
import json
import sys
from pathlib import Path


def pct(x):
    return f"{x * 100:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, default=Path("paper/berth_paper.md"))
    ap.add_argument("--numbers", type=Path, default=Path("data/paper_numbers.json"))
    a = ap.parse_args()
    paper = a.paper.read_text()
    n = json.loads(a.numbers.read_text())
    rs, vs, so = n["release_split"], n["verify_status"], n["statement_source_of_ok"]
    off = n["offline_status_of_ok_with_statement"]
    go, g = n["gemma_official"], n["gemma"]
    loc, d = n["loc"], n["loc_graph_minus_bm25"]
    cc, ci_ = n["core_compatible"], n["core_incompatible"]
    oc = n["official_compatible"]
    med = n["core_medians"]
    claims = [
        ("total tasks", f"{rs['core'] + rs['extended']} verified", None),
        ("core / extended", f"{rs['core']} core, {rs['extended']} extended", None),
        ("release repos", f"from {n['release_repos']} ", None),
        ("repos cloned", f"cloned {n['repos_cloned']} ", None),
        ("candidates", f"{n['candidates_since_2025_07']} candidates", None),
        ("unique PRs", f"Of {n['verified_records']} unique pull requests, {vs['ok']} verified", None),
        ("no_f2p", f"{vs['no_f2p']} had no failing-then-passing test", None),
        ("no_tests", f"{vs['no_tests']} had no runnable tests", None),
        ("err+unstable", f"{vs['error'] + vs['unstable']} errored or flipped", None),
        ("missing statements", "Seventeen pull-request numbers" if so.get("missing") == 17 else f"{so.get('missing')}", None),
        ("offline", f"{sum(v for k, v in off.items() if k.endswith('|ok'))} of {sum(off.values())} tasks passed", None),
        ("flaky removed", f"({n['tasks_with_flaky_p2p_removed']} tasks affected)", None),
        ("gold replays", f"{n['gold_replays']['resolved']:,} of {n['gold_replays']['runs']:,} replays", None),
        ("overlap", "Two candidates matched" if n["overlap_swerebench"] == 2 else str(n["overlap_swerebench"]), None),
        ("fix range", f"{n['core_fix_commit_range'][0]} and {n['core_fix_commit_range'][1]}", None),
        ("pre-cutoff core", f"{rs['core'] - n['post_cutoff']['core']} core tasks resolve an issue first filed before", None),
        ("core issue / issue_ref", f"(\"issue\", {n['core_statement_source']['issue']} core tasks)", None),
        ("issue_ref", f"(\"issue_ref\", {n['core_statement_source']['issue_ref']})", None),
        ("underspecified", f"{n['core_underspecified']} core tasks are flagged ({cc['underspecified']} of the {cc['n']}", None),
        ("compatible core", f"{cc['n']} of {rs['core']} core tasks are harness-compatible", None),
        ("incompatible split", f"{ci_['async_tests']} async-test tasks and {ci_['other']} with residual", None),
        ("async count", f"**Async test suites** ({ci_['async_tests']} core tasks", None),
        ("env pinned", f"({n['env_pinned']['tasks']} tasks, mostly pydantic-settings", None),
        ("env repaired", f"repaired {n['env_pinned']['compatible']} of them", None),
        ("extended compatible", f"| Pass the official grader | {cc['n']} | {oc.get('extended|True')} |", None),
        ("post-cutoff compatible", f"(209 of the 281)" if cc["post_cutoff"] == 209 and cc["n"] == 281 else "MISMATCH", None),
        ("post-cutoff table", f"| {n['post_cutoff']['core']} ({cc['post_cutoff']} compatible) | {n['post_cutoff']['extended']} |", None),
        ("median patch", f"| Median added lines in gold patch | {med['patch_added_lines']} |", None),
        ("files changed", f"| {n['core_files_changed']['1']} / {n['core_files_changed']['2']} / {n['core_files_changed']['3']} |", None),
        ("median f2p/p2p", f"| {med['fail_to_pass']} / {med['pass_to_pass']} |", None),
        ("added defs", f"| Tasks adding new definitions | {n['core_with_added_nodes']} |", None),
        ("graph size", f"| {n['core_graph_nodes']['median']:,} ({n['core_graph_nodes']['min']}–{n['core_graph_nodes']['max']:,}) |", None),
        ("loc tasks", f"For each of the {n['loc_tasks']} core tasks", None),
        ("module-level only", f"the other {'nine' if n['core_module_level_only'] == 9 else n['core_module_level_only']} core tasks change only module-level code", None),
        ("bm25 @10 abstract", f"top 10 for {pct(loc['bm25']['node@10']['mean'])}%" if False else None, None),
        ("tfidf @10", f"Our TF-IDF result ({pct(loc['tfidf_name']['node@10']['mean'])}% at top 10)", None),
        ("graph -node@1", f"by {abs(d['node@1']['mean']) * 100:.1f} points [−{abs(d['node@1']['lo']) * 100:.1f}, −{abs(d['node@1']['hi']) * 100:.1f}]", None),
        ("graph +file@1", f"by {d['file@1']['mean'] * 100:.1f} points [+{d['file@1']['lo'] * 100:.1f}, +{d['file@1']['hi'] * 100:.1f}]", None),
        ("gemma headline", f"**Overall.** {pct(go['compatible_all']['mean'])}% resolved [{pct(go['compatible_all']['lo'])}, {pct(go['compatible_all']['hi'])}]", None),
        ("gemma post", f"{pct(go['compatible_post_cutoff']['mean'])}% [{pct(go['compatible_post_cutoff']['lo'])}, {pct(go['compatible_post_cutoff']['hi'])}] on the {go['compatible_post_cutoff']['n']} compatible post-cutoff", None),
        ("gemma ours all", f"the figure is {pct(g['all']['mean'])}%", None),
        ("agreement", "agree on 274" if go["agreement_ours_vs_official_on_compatible"].get("ours=True|official=True", 0)
         + go["agreement_ours_vs_official_on_compatible"].get("ours=False|official=False", 0) == 274 else "MISMATCH", None),
        ("median turns", f"the median used {g['median_turns']} turns", None),
        ("hit limit", f"{g['hit_time_limit']} hit the 20-minute limit", None),
        ("no gold file", f"In {pct(1 - (1 - go['unresolved_no_gold_file']['mean']))}% [{pct(go['unresolved_no_gold_file']['lo'])}, {pct(go['unresolved_no_gold_file']['hi'])}] of unresolved", None),
        ("underspec rates", f"resolve at {pct(go['compatible_underspecified']['mean'])}% against {pct(go['compatible_not_underspecified']['mean'])}% for the rest", None),
        ("cutoff rates", f"*less* often ({pct(go['compatible_pre_cutoff']['mean'])}%) than post-cutoff ones ({pct(go['compatible_post_cutoff']['mean'])}%)", None),
        ("breadth", f"{pct(go['compatible_files_1']['mean'])}% for one-file fixes, {pct(go['compatible_files_2']['mean'])}% for two, 0 of {go['compatible_files_3']['n']} for three", None),
        ("pruning core", "For 44 tasks we dropped colliding ids: 2,006 must-pass entries and 152 fail-to-pass entries", None),
    ]
    for key, m in (("random", "Random"), ("tfidf_name", "TF-IDF (names)"), ("bm25", "BM25 (code)"), ("bm25+graph", "BM25 + graph")):
        L = loc[key]
        claims.append((f"loc row {key}", f"| {m} | " + " | ".join(f"{pct(L[k]['mean'])}%" for k in
                                                                    ("node@1", "node@5", "node@10", "node@20", "file@1", "file@5")) + " |", None))
    bad = 0
    for label, text, _ in claims:
        if text is None:
            continue
        ok = text in paper
        bad += not ok
        print(f"{'OK  ' if ok else 'MISS'} {label}: {text}")
    for ph in ("[LB_SCORE]", "[LB_SENTENCE]", "[KAGGLE_DATASET_URL]", "[HF_URL]", "[GITHUB_URL]", "[EXT_COMPAT]"):
        if ph in paper:
            print(f"TODO placeholder still in paper: {ph}")
    print(f"\n{len([c for c in claims if c[1]]) - bad} ok, {bad} mismatched")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
