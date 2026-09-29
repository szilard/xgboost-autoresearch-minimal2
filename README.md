## Optimizing XGBoost Machine Learning Models with AI Agents

This is a follow-up to [xgboost-autoresearch](https://github.com/szilard/xgboost-autoresearch).

It provides a "minimal" building block that can be used by an orchestrator such as
[xgboost-autoresearch-minimal2-runs](https://github.com/szilard/xgboost-autoresearch-minimal2-runs)
to run repeated trials of XGBoost tuning with various agents/LLMs.

An AI agent (Claude Code, Codex, ...) autonomously improves an XGBoost model for a fixed 2-hour budget, following the instructions in `program.md`:

- **Task:** predict whether a flight departs 15+ minutes late (2005 airline data, balanced, 200K train / 50K eval / 50K holdout rows), measured by AUC.
- **Loop:** the agent edits `train.py` (data prep, feature engineering, hyperparameters, model training), commits, runs it via `harness.py`, and keeps the commit if eval AUC improved or resets it otherwise. It also researches ideas on the web and logs every experiment in `results.tsv` and `research-log.md`.
- **Guardrails:** `harness.py` enforces time limits (1 min training, 5 min evaluation per run) and scores the saved model + feature code row by row, so features that aggregate over the dataset don't work. The holdout set and the scripts that score it are off-limits to the agent.
- **Ground truth:** after the run, the human scores every kept model on the holdout set to check that the eval AUC gains generalize (`groundtruth_all.tsv`, `auc_history.png`).

See [README-autoresearch.md](README-autoresearch.md) for setup and details, and `results/` for archived runs.

Recommended machine (training XGBoost): m8i.2xlarge (8 cores, 32GB RAM)

