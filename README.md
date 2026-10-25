# Berth

Harness-compatible, post-cutoff Python tasks for offline Gemma 4 coding agents. This is the pipeline code and paper
for the **Gemma 4 Developer Agent Paper Track** (Kaggle, 2026).

- **Dataset:** https://huggingface.co/datasets/JayPatel2574/berth (also on Kaggle: [KAGGLE_DATASET_URL])
- **Paper:** [`paper/berth_paper.pdf`](paper/berth_paper.pdf) ([Markdown](paper/berth_paper.md))

Berth has 653 verified bug-fix and feature tasks (301 core, 350 extended, 2 overlap) from 24 permissively licensed
Python libraries. They are packaged in the competition harness's data layout (`tasks.jsonl`, history-free snapshots,
official-format code graphs) and install and test offline from a bundled wheel pool. 281 core tasks also pass the
harness's own Phase 2 grading rules, reproduced with its code.

## Pipeline (`scripts/`)

| Step | Script | What it does |
|---|---|---|
| 1 | `mine_candidates.py` | First-parent commits naming a PR that change 1–3 source files and at least one test file |
| 2 | `verify_task.py` (`run_queue.sh`) | Python 3.13 venv; tests fail before / pass after the fix, twice; `--pin-to-commit-date` resolves dependencies as of the fix date |
| 3 | `localize.py` | Gold patch → official graph node ids (`nodes_edited`, `nodes_added`) |
| 4 | `fetch_statements.py`, `gh_fetch_local.py` | Issue / PR text from GitHub (closing links, keywords, earlier mentions) |
| 5 | `package_task.py` (`run_package.sh`) | Snapshot, graph, stand-in embeddings, wheels; offline re-check in a network-less namespace |
| 6 | `run_gemma.py --model gold`, `stability.py` (`run_stability.sh`) | Three gold replays under parallel load; flaky must-pass tests removed |
| 7 | `official_grade_check.py` | Official swegemma Phase 2 rules (its own setup and JUnit validator), gold and empty patch; validator-collision pruning |
| 8 | `check_overlap.py`, `quality_flags.py`, `export_release.py` | SWE-rebench decontamination, underspecification flag, release `tasks.jsonl` |
| 9 | `loc_baselines.py` | Random / TF-IDF / BM25 / BM25 + graph localization |
| 10 | `run_gemma.py`, `build_kaggle_bundle.py`, `build_kaggle_notebook.py` | Reference agent on Kaggle 4×L4 (vLLM, competition checkpoint); `official_regrade_runs.py` re-grades its patches |
| 11 | `paper_numbers.py`, `paper_figures.py`, `build_pdf.py` | Every number and figure in the paper, recomputed from data files |
| — | `build_release_dir.py` | Public dataset folder (Hugging Face / Kaggle layouts) |

`scripts/third_party/` holds `code_graph.py` and `random_embeddings.py`, copied unchanged from
[nakranivaibhav/gemma-4-developer-agent](https://github.com/nakranivaibhav/gemma-4-developer-agent) (Apache-2.0).
That code-graph builder reproduces all 127 official graphs.

`competition_agent/` is the reference agent in the competition's declarative format, as submitted to the main
competition's leaderboard.

## Reproducing

Requirements: Linux, Python 3.13, [`uv`](https://github.com/astral-sh/uv), `git`, `unshare` (for offline checks), and
the competition's `swegemma`/`adk_submission` wheels for step 7. GitHub API access is needed for step 4, and a Kaggle
account with the main competition joined for step 10.

```bash
python -I scripts/mine_candidates.py repos --since 2023-01-01 > data/candidates.jsonl
scripts/run_queue.sh data/queue_recent.txt data/verify          # step 2 (resumable)
python -I scripts/localize.py data/verify
python -I scripts/fetch_statements.py queue data/verify && python3 scripts/gh_fetch_local.py \
  && python -I scripts/fetch_statements.py merge data/verify
scripts/run_package.sh data/pkg_core.txt                          # step 5
scripts/run_stability.sh && python -I scripts/stability.py data/verify runs/stability
unshare -rn <venv with swegemma>/bin/python -I scripts/official_grade_check.py --release release \
  --python <python3.13> --uv uv --wheels release/wheels
python -I scripts/export_release.py data/verify --release release --require-stability
python -I scripts/paper_numbers.py && python -I scripts/paper_figures.py --out paper/figures
```

## Licence

Code: Apache-2.0 (see `LICENSE`). Dataset snapshots keep their source repositories' licences (MIT, BSD, Apache-2.0).

## Citation

Jay Patel. *Berth: Harness-Compatible, Post-Cutoff Python Tasks for Offline Gemma 4 Coding Agents.* Gemma 4 Developer
Agent Paper Track, Kaggle, 2026.
