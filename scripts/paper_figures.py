"""Figures for the Berth paper, drawn from data/paper_numbers.json (PNG at 200 dpi, white surface).

fig1_funnel.png        construction funnel (one series)
fig2_gemma.png         Gemma 4 31B resolve rate by subset under the official grader, harness-compatible core tasks
fig3_localization.png  node hit@k for four rankers (four series, direct-labelled + distinct markers)

Palette: the dataviz skill's validated reference categorical slots 1-4 (light mode); grid and text in recessive ink.

usage: python -I paper_figures.py --out paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def tidy(ax, grid_axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def funnel(n, out):
    st = n["verify_status"]
    so = n["statement_source_of_ok"]
    off = n["offline_status_of_ok_with_statement"]
    with_stmt = n["verify_status"]["ok"] - so.get("missing", 0)
    offline_ok = sum(v for k, v in off.items() if k.endswith("|ok"))
    stable = n["stability_status"]["ok"]
    rs = n["release_split"]
    stages = [
        ("Candidate commits (Jul 2025 – Oct 2026)", n["candidates_since_2025_07"]),
        ("Verified: tests fail before, pass after (×2)", st["ok"]),
        ("Problem statement fetched", with_stmt),
        ("Offline check: installs from bundled wheels", offline_ok),
        ("Stable over 3 gold replays", stable),
        ("Released: core (real issue)", rs["core"]),
        ("Core passing the official grader", n["core_compatible"]["n"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    y = list(range(len(stages)))[::-1]
    ax.barh(y, [v for _, v in stages], color=SERIES[0], height=0.62)
    for yi, (lab, v) in zip(y, stages):
        ax.text(v + 12, yi, f"{v}", va="center", color=INK, fontsize=9.5, fontweight="bold")
    ax.set_yticks(y, [lab for lab, _ in stages])
    ax.set_xlim(0, max(v for _, v in stages) * 1.12)
    ax.set_xlabel("tasks")
    fig.suptitle(f"Construction funnel ({rs['extended']} more tasks go to the extended split)", x=0.015, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def gemma(n, out):
    g = n["gemma_official"]
    rows = [
        ("All (harness-compatible)", g["compatible_all"]),
        ("Issue & PR after cutoff", g["compatible_post_cutoff"]),
        ("Issue before cutoff", g["compatible_pre_cutoff"]),
        ("Not underspecified", g["compatible_not_underspecified"]),
        ("Underspecified (flagged)", g["compatible_underspecified"]),
        ("Fix touches 1 file", g["compatible_files_1"]),
        ("Fix touches 2 files", g["compatible_files_2"]),
        ("Fix touches 3 files", g["compatible_files_3"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, r) in zip(y, rows):
        ax.plot([r["lo"] * 100, r["hi"] * 100], [yi, yi], color=SERIES[0], linewidth=2, solid_capstyle="round")
        ax.plot(r["mean"] * 100, yi, "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(max(r["hi"] * 100, r["mean"] * 100) + 1.2, yi, f"{r['mean'] * 100:.1f}%  (n={r['n']})",
                va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.set_xlim(-1, 45)
    ax.set_xlabel("resolved (%), 95% bootstrap interval")
    fig.suptitle(f"Gemma 4 31B, official grading ({g['compatible_all']['n']} compatible core tasks)",
                 x=0.015, ha="left", fontsize=11, fontweight="bold", color=INK)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def localization(n, out):
    names = [("random", "Random"), ("tfidf_name", "TF-IDF on symbol names"), ("bm25", "BM25 on code"),
             ("bm25+graph", "BM25 + graph propagation")]
    ks = [1, 5, 10, 20]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (key, lab) in enumerate(names):
        ys = [n["loc"][key][f"node@{k}"]["mean"] * 100 for k in ks]
        ax.plot(ks, ys, color=SERIES[i], linewidth=2, marker=markers[i], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        ax.text(20.8, ys[-1] + (3.2 if key == "bm25+graph" else -3.2 if key == "bm25" else 0), lab, va="center",
                color=INK2, fontsize=8.5)
    ax.set_xticks(ks, [f"top {k}" for k in ks])
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.set_ylabel("a gold function in the top k (%)")
    ax.set_title(f"Function-level localization on {n['loc_tasks']} core tasks (official-format graphs)", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(ax, "y")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", type=Path, default=Path("data/paper_numbers.json"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = json.loads(a.numbers.read_text())
    funnel(n, a.out / "fig1_funnel.png")
    gemma(n, a.out / "fig2_gemma.png")
    localization(n, a.out / "fig3_localization.png")
    print("wrote", sorted(p.name for p in a.out.glob("*.png")))


if __name__ == "__main__":
    main()
