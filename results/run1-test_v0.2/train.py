import numpy as np
import pandas as pd
import time
import xgboost as xgb
from pathlib import Path
from harness import save_and_evaluate


data_dir = Path(__file__).parent / "data"
train = pd.read_csv(f"{data_dir}/train.csv")

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["CRSDepTime", "Distance"]
target   = "dep_delayed_15min"


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

# target encoding on disjoint parts: model h is fit on all parts but h, with tables fit on part h,
# so no training row's features contain its own (or its group-mates' in-sample) labels
from sklearn.model_selection import StratifiedKFold, train_test_split
n_parts = 3
parts = [(fit_idx, te_idx) for seed in range(3) for fit_idx, te_idx in
         StratifiedKFold(n_parts, shuffle=True, random_state=seed).split(train, train[target])]
te_smooth = 20
te_keys = {
    "DateOrigin": ["Doy", "Origin"],
    "DateDest": ["Doy", "Dest"],
    "DateCarrier": ["Month", "DayofMonth", "UniqueCarrier"],
    "Origin": ["Origin"],
    "Dest": ["Dest"],
    "Route": ["Origin", "Dest"],
    "CarrierOrigin": ["UniqueCarrier", "Origin"],
    "OriginHour": ["Origin", "Hour"],
    "DestHour": ["Dest", "Hour"],
    "RouteHour": ["Origin", "Dest", "Hour"],
    "Flight": ["UniqueCarrier", "Origin", "Dest", "CRSDepTime"],
    "CarrierOriginHour": ["UniqueCarrier", "Origin", "Hour"],
}
# neighbouring days at the same airport (disruptions persist): looked up in the date tables
te_shifted = {
    "PrevDateOrigin": ("DateOrigin", ["DoyPrev", "Origin"]),
    "NextDateOrigin": ("DateOrigin", ["DoyNext", "Origin"]),
    "PrevDateDest": ("DateDest", ["DoyPrev", "Dest"]),
    "NextDateDest": ("DateDest", ["DoyNext", "Dest"]),
}
te_names = list(te_keys) + list(te_shifted)
month_start = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]  # 2005, by month

def te_keys_of(df):
    """(table name, key strings) per TE feature, built with plain Python (fast for single rows)."""
    str_cols = {col: df[col].astype(str).tolist() for col in cat_cols}
    str_cols["Hour"] = (df["CRSDepTime"] // 100).astype(str).tolist()
    str_cols["CRSDepTime"] = df["CRSDepTime"].astype(str).tolist()
    doy = [month_start[int(m[2:])] + int(d[2:]) for m, d in zip(str_cols["Month"], str_cols["DayofMonth"])]
    for col, shift in [("Doy", 0), ("DoyPrev", -1), ("DoyNext", 1)]:
        str_cols[col] = [str(v + shift) for v in doy]
    specs = {name: (name, cols) for name, cols in te_keys.items()} | te_shifted
    return {name: (table, ["|".join(v) for v in zip(*(str_cols[c] for c in cols))])
            for name, (table, cols) in specs.items()}

def fit_tables(d):
    y = (d[target] == "Y").astype(float)
    prior = y.mean()
    tables = {}
    for name, (_, key) in te_keys_of(d).items():
        if name not in te_keys:
            continue
        g = y.groupby(np.array(key))
        tables[name] = ((g.sum() + prior * te_smooth) / (g.count() + te_smooth)).to_dict()
    return tables

te_tables = [fit_tables(train.iloc[te_idx]) for _, te_idx in parts]

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    keys = te_keys_of(df)
    te = {f"{name}_{h}": [tables[table].get(k, np.nan) for k in key]
          for h, tables in enumerate(te_tables) for name, (table, key) in keys.items()}
    X = pd.concat([X, pd.DataFrame(te, index=X.index, dtype=float)], axis=1)
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

def view(X, h):
    """Base features plus the encodings from the tables of part h, under common names."""
    base = [c for c in num_cols + cat_cols if c not in ("Origin", "Dest")]
    return X[base + [f"{name}_{h}" for name in te_names]].rename(
        columns={f"{name}_{h}": name for name in te_names})

class PartsEnsemble:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(view(X, h)) for h, m in enumerate(self.models)], axis=0)


t0 = time.time()
models = []
X_train, y_train = prepare(train)
for h, (fit_idx, _) in enumerate(parts):
    y_fit = y_train[fit_idx]
    X_tr, X_va, y_tr, y_va = train_test_split(view(X_train.iloc[fit_idx], h), y_fit, test_size=0.1, random_state=42, stratify=y_fit)
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=10,
        learning_rate=0.05,
        tree_method="hist",
        subsample=0.8,
        min_child_weight=5,
        max_bin=64,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=100,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"Model {h}: best iteration {m.best_iteration}, val AUC {m.best_score:.4f}")
    models.append(m)
model = PartsEnsemble(models)
print(f"Training time: {time.time() - t0:.1f}s")


# Saves {model, prepare} to artifacts/ and scores eval.csv row by row. Keep this call last.
save_and_evaluate(model, prepare)
