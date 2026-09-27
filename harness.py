"""Save the trained artifact and score it row by row. Not modified by the agent.

The artifact is `{"model": model, "prepare": prepare}` pickled with cloudpickle,
which stores `prepare` by value together with every module-level lookup it uses
(cat_levels etc.), so it can be scored later without train.py or the training data.
Artifacts live in the gitignored artifacts/ folder, named by full commit hash.
"""
import os
import pickle
import subprocess
import time
import multiprocessing as mp
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

repo_dir = Path(__file__).parent
data_dir = repo_dir / "data"
artifacts_dir = repo_dir / "artifacts"
n_workers = os.cpu_count()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=repo_dir, text=True).strip()


def find_artifact(commit):
    """Artifact path for a full or abbreviated commit hash."""
    matches = sorted(artifacts_dir.glob(f"{commit}*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected 1 artifact for {commit}, found {len(matches)}")
    return matches[0]


def load_artifact(commit):
    with open(find_artifact(commit), "rb") as f:
        return pickle.load(f)


_prepare = None


def _prepare_rows(df):
    """prepare() each row on its own; runs in a worker process."""
    out = [_prepare(df.iloc[[i]]) for i in range(len(df))]
    return pd.concat([X for X, _ in out]), np.concatenate([y for _, y in out])


def score_by_row(artifact, df):
    """prepare() row by row across processes, then predict in one batch. Returns AUC."""
    global _prepare
    _prepare = artifact["prepare"]
    per_worker = -(-len(df) // n_workers)
    chunks = [df.iloc[i:i + per_worker] for i in range(0, len(df), per_worker)]
    # fork so the workers inherit _prepare as set above
    with mp.get_context("fork").Pool(n_workers) as pool:
        out = pool.map(_prepare_rows, chunks)
    X = pd.concat([X for X, _ in out])
    y = np.concatenate([y for _, y in out])
    y_prob = artifact["model"].predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)


def save_and_evaluate(model, prepare):
    """Save the artifact for the current commit, then score eval.csv with the reloaded copy."""
    blob = cloudpickle.dumps({"model": model, "prepare": prepare})
    # score exactly what the ground truth evaluation will load, not the in-memory objects
    artifact = pickle.loads(blob)

    if git("status", "--porcelain", "--", "train.py"):
        print("WARNING: train.py has uncommitted changes, artifact not saved")
    else:
        artifacts_dir.mkdir(exist_ok=True)
        path = artifacts_dir / f"{git('rev-parse', 'HEAD')}.pkl"
        path.write_bytes(blob)
        print(f"Artifact: {path.relative_to(repo_dir)} ({len(blob) / 1e6:.1f} MB)")

    eval_df = pd.read_csv(data_dir / "eval.csv")
    t0 = time.time()
    eval_auc = score_by_row(artifact, eval_df)
    print(f"Eval time: {time.time() - t0:.1f}s")
    print(f"Eval AUC: {eval_auc:.4f}")
