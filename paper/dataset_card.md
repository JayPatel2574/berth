---
license: apache-2.0
pretty_name: Berth
language:
- en
tags:
- code
- software-engineering
- agents
- swe-bench
- gemma
size_categories:
- n<1K
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: core
    path: core.jsonl
  - split: extended
    path: extended.jsonl
  - split: overlap
    path: overlap.jsonl
---

# Berth: harness-compatible, post-cutoff Python tasks for offline Gemma 4 coding agents

Berth is a set of **653 verified bug-fix and feature tasks** from 24 permissively licensed Python libraries. Every task is
packaged in the data layout of the [Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)
harness and installs and tests **with no network**. The layout covers `tasks.jsonl` columns, history-free repository
snapshots, and code graphs in the official format.

It is meant as held-out validation and as a training pool for agents built for that harness. The competition's public
development set has 129 tasks from 4 repositories.

| Split | Tasks | Pass the official grader | Statement | Use |
|---|---|---|---|---|
| `core` | 301 | 281 | real GitHub issue | evaluation |
| `extended` | 350 | 297 | pull-request description | training |
| `overlap` | 2 | 2 | real issue | also in SWE-rebench; kept apart |

**Recommended headline subset:** `core` tasks with `harness_compatible == true` and `post_cutoff == true` (209 tasks).
In these, both the issue and the pull request postdate Gemma 4's January 2025 pretraining cutoff, and the task passes the
competition's own grading rules.

## How every task was checked

1. **Fail before, pass after**: in a fresh Python 3.13 environment, tests touched by the fix fail without it and pass
   with it, identically in two runs.
2. **Offline**: inside a network-less namespace, the environment installs only from `wheels/`; the empty patch fails
   `FAIL_TO_PASS` and the gold patch passes.
3. **Stable**: three more gold replays through an agent runner's tools and grader, under parallel load.
4. **Official grader** (`harness_compatible`): the harness's own Phase 2 sequence is reproduced offline, with
   `setup_workspace_test_config` and `_validate_junit_xml` imported from the competition's `swegemma` wheel. The whole
   target test files must pass with `-p no:anyio`, and required tests are matched by name prefix. The gold patch must
   resolve and the empty patch must not.

Tasks failing step 4 are kept and flagged. Most use async test suites, which the official grader cannot run. To
satisfy the official validator's prefix matching, some test ids were removed; they are listed per task in
`FAIL_TO_PASS_dropped_validator` and `PASS_TO_PASS_dropped_validator`, and every task keeps at least one
`FAIL_TO_PASS` test. For 33 tasks, dependencies were resolved as of the fix commit's date
(`env_pinned_to_commit_date`).

## Files

| Path | Content |
|---|---|
| `tasks.jsonl` (+ `core/extended/overlap.jsonl`) | One task per line (fields below) |
| `snapshots/<instance_id>.tgz` | Repository at `base_commit` as a fresh one-commit git repo (no forward history) |
| `graphs/<repo>_<commit>.json` | Code graph, official node-link format (built with an Apache-2.0 builder that reproduces all 127 official graphs) |
| `embeddings/<repo>_<commit>.npz` | 256-d **stand-in** vectors (deterministic random; the official encoder is unpublished) |
| `wheels/` | Every wheel any task needs, CPython 3.13 / manylinux x86_64 |
| `runs/gemma-4-31b-it-qat-w4a16-ct/` | Reference-agent trajectories (`.traj.jsonl`), our grading (`.json`), official grading (`.official.json`) |
| `results/` | Localization baseline per task; every number in the paper (`paper_numbers.json`) |

On Kaggle, `snapshots/` and `wheels/` ship as `snapshots.tar` and `wheels.tar`, because Kaggle unpacks archives inside
datasets. Run `tar -xf snapshots.tar && tar -xf wheels.tar` first.

## Task fields

Official columns: `instance_id, repo, base_commit, patch, test_patch, problem_statement, hints_text, created_at,
FAIL_TO_PASS, PASS_TO_PASS`.

Berth fields:
- `split`, `statement_source` (`issue` / `issue_ref` / `pr`), `issue_urls`, `pr_url`, `fix_commit`, `merged_at`,
  `issue_created_at`
- `post_cutoff`: issue and PR both created on or after 2025-02-01
- `harness_compatible`, `env_pinned_to_commit_date`
- `PASS_TO_PASS_flaky`, `FAIL_TO_PASS_dropped_validator`, `PASS_TO_PASS_dropped_validator`
- `localization`: `files`, `nodes_edited` and `nodes_added` (official graph node ids), `module_level`, `aux_files`
- `quality`: `underspecified` (the hidden tests call a name the fix introduces and the issue never mentions),
  `new_api_in_tests`, `f2p_count`, `patch_lines`, `files_changed`, `statement_chars`
- `requirements`, `build_requires`, `graph`, `embeddings`, `snapshot`, `test_patch_files`, `patch_files`

## Baselines (core)

- **Localization** (292 tasks whose fix edits a function or class). BM25 puts a gold function in its top 10 for 64.0%
  of tasks, against 9.9% for random. Graph propagation trades top-1 function accuracy (−6.9 points) for top-1 file
  accuracy (+6.5).
- **Gemma 4 31B** (`gemma-4-31b-it-qat-w4a16-ct` on Kaggle 4×L4, simple single agent, 20-minute budget, official
  grading): 19.6% of the 281 compatible tasks, and 22.0% of the 209 post-cutoff ones. In 59% of failures the patch
  touched none of the files the real fix changes.

## Licence

Each snapshot keeps the licence of its source repository (MIT, BSD or Apache-2.0; see `LICENSES/`). Berth's
annotations, results and pipeline code are released under Apache-2.0. Pipeline code:
https://github.com/JayPatel2574/berth

## Citation

Jay Patel. *Berth: Harness-Compatible, Post-Cutoff Python Tasks for Offline Gemma 4 Coding Agents.* Gemma 4 Developer
Agent Paper Track, Kaggle, 2026.
