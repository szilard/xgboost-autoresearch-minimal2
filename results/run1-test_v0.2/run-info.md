# Run run1-test_v0.2

| | |
|---|---|
| Agent | Claude Code 2.1.283, on the host (no container), permission mode `auto` |
| Model | claude-opus-5-5 (Opus 5.5) |
| Reasoning effort | medium |
| Upstream commit | fdbad98 (main, v0.2: scheduled CRSDepTime, pandas prepare from S3) |
| Run tag / branch | sep27b |
| Date | 2026-09-27 (clock 22:21:46 -> 2026-09-28 00:01:57 UTC, 1h40m11s) |
| Session id | 6dc90c0a-b6a8-4eda-95a2-a9271e5d0cbe |

Identified afterwards from the Claude Code session log
(`~/.claude/projects/-home-ubuntu-xgboost-autoresearch-minimal2/6dc90c0a-....jsonl`):
the session contains the `harness.py run` calls and outputs of this run, and its
timestamps match `timing/clock.json`.

## Not a clean run

This was a test run in the same session as development work, not a fresh agent:

- Earlier in the session (branch `sep27`, 20:18-20:29 UTC) the agent ran a first
  experiment on the previous data, which leaked the target through the actual
  `DepTime`; it was interrupted and wrapped up. The research log refers to it
  ("prior knowledge from this session's earlier run on the leaky data").
- Between the two runs, the same agent rewrote `prepare.py` and made a pass over
  all files, so its context already contained the data split, the holdout path and
  the ground truth scripts that `program.md` puts off-limits.

The agent stopped on its own after 59 experiments, ~20 min before the 2-hour budget.
Results are not directly comparable to runs started in a fresh session/container.

## Results

- Runs: 60 (baseline + 59 experiments): 25 keep (incl. baseline), 34 discard, 1 crash
- Best Eval AUC: **0.7717** at `8c8b36b`, baseline 0.7203
- Holdout AUC of `8c8b36b`: **0.7663** (gap holdout - eval = -0.0054)

Final model: bag of 9 XGBoost models (3 disjoint parts x 3 split seeds), each fit on
2/3 of train with target encodings (date x origin/dest/carrier, neighbouring days,
route, airport x hour, scheduled flight, ...) from tables fit on the other 1/3.
Most of the gain comes from the date x airport encodings, which work because
train/eval/holdout are a random split sharing the same days.
