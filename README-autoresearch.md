# autoresearch XGBoost

an adaptation of A. Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) project to XGBoost

The idea: give an AI agent a small but real XGBoost training setup and let it experiment autonomously overnight. It modifies the code, trains, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a small, single-file XGBoost setup. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` markdown files that provide context to the AI agents and set up your autonomous research org. 

## How it works

The repo is deliberately kept small:

- **`prepare.py`** - builds train/eval/holdout.csv from data/2005.csv. Human only; the AI agent must not read it.
- **`train.py`** - the single file the agent edits. Contains the XGBoost model training. Everything is fair game that will lead to a model that generalizes on unseen data: data preparation, feature engineering, choosing hyperparameters, and model training. **This file is edited and iterated on by the agent**.
- **`program.md`** - baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.
- **`harness.py`** - saves the trained model and `prepare` to `artifacts/<commit>.pkl` (gitignored) and scores `eval.csv` row by row. Not modified by the AI agent.
- **`check_groundtruth.py`** - script to check the "ground truth" AUC on `holdout.csv` by the human, from the saved artifact (no retraining): `python3 check_groundtruth.py [commit]`. AI should not access this file.
- **`run_groundtruth_all.sh`** + **`plot_auc_history.py`** - after a run, score every kept experiment in `results.tsv` on the holdout set (`groundtruth_all.tsv`) and plot eval vs holdout AUC (`auc_history.png`). Human only.

## Quick start

```bash

# 1. Install dependencies
pip install pandas polars xgboost scikit-learn cloudpickle matplotlib --break-system-packages

# 2. Build train/eval/holdout.csv from data/2005.csv
python3 prepare.py

# 3. Manually run a single training experiment (to verify everything works)
python3 train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py             - builds the data splits (human only)
train.py               - XGBoost training (AI agent modifies this)
harness.py             - saves the artifact, row-by-row eval scoring
check_groundtruth.py   - holdout scoring of a saved artifact (human only)
run_groundtruth_all.sh - holdout scoring of all kept experiments (human only)
plot_auc_history.py    - plot of eval vs holdout AUC (human only)
program.md             - agent instructions
data/                  - 2005.csv source and train/eval/holdout.csv splits (gitignored)
artifacts/             - saved model + prepare per commit (gitignored)
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Self-contained.** No external dependencies beyond XGBoost, pandas, scikit-learn, cloudpickle, polars (data prep) and matplotlib (plot). No distributed training, no complex configs. 
