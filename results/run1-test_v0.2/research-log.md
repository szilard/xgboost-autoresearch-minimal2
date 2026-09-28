# Research log — run sep27b

Rows here map to results.tsv (commit hashes). Data: the scheduled departure time CRSDepTime (no actual DepTime, so no target leak).

## Exp 1 — fdbad98 — baseline
Starter train.py: 30 trees, depth 6, lr 0.1, native categoricals. Eval AUC 0.7203. Train 0.3s, eval ~31s.
Prior knowledge from this session's earlier run on the leaky data: early stopping + many more trees + deeper trees gave big gains, so start there.

## Research (before exp 2)
- XGBoost categorical docs (https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html): max_cat_to_onehot / max_cat_threshold for high-cardinality Origin/Dest.
- Tuning guides (https://xgboost.org/code/hyperparameter-tuning-guide/, https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/): tune depth + min_child_weight, then subsample/colsample (~0.8), then lambda/alpha.
- Flight delay literature (e.g. https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches): origin airport and carrier matter most; time-of-day and date/weather effects (cascading delays late in the day).

## Exp 2 — 19e81e4 — early stopping + depth 8 + lr 0.05 (follow-up)
Eval 0.7321 (+0.012). best_it only 237 (val 0.7396) → overfits quickly; regularisation should help.

## Exp 3 — f850823 — subsample/colsample 0.8 (follow-up, from tuning guide)
Eval 0.7393 (+0.0072), val 0.7460. Randomisation helps a lot → model is variance-limited. Hypothesis: native categorical partition splits on Origin/Dest (~300 levels) overfit; limit them.

## Exp 4 — 7ee3a70 — max_cat_threshold 16 (follow-up) → keep
Eval 0.7419 (+0.0026), best_it 507 (was 230): fewer categories per split = less overfitting, more trees useful.

## Exp 5 — 9d74daa — max_cat_threshold 4 (follow-up) → keep
Eval 0.7450 (+0.0031), best_it 894, val 0.7533. Trend continues; try the extreme (1 = one category vs rest).

## Exp 6 — 4622093 — max_cat_threshold 1 → discard
Eval 0.6646 (collapse). With threshold 1 the categorical splits are nearly useless. Sweet spot is small but >1 (4 best so far).

## Research (before exp 7)
- Target encoding needs cross-fitting to avoid label leakage (sklearn TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html). Folds here come from a hash of the row's features so the encoding is identical for single-row eval.

## Exp 7 — ce3aca1 — OOF target encoding date x origin/dest/carrier (exploration) → discard
Val AUC 0.7682 (!) but Eval 0.7147 (-0.030). Debugged: fold hash stable single-row vs batch; artifact scored row-by-row on val rows reproduces 0.7682 → pipeline consistent, gap is statistical. Suspect: val rows' labels are inside the TE tables (tables fit on all of train), so val is not an honest holdout for TE features. Keys date x origin have ~2 rows each → encodings very noisy.

## Exp 8 — cef442f — OOF TE, tables fit on fit split only (diagnostic) → discard
Honest val 0.7270, best_it 14, Eval 0.7201. Confirms exp 7's val was inflated and that OOF TE hurts here.
Why: the trees can identify a date x origin group via the Month/DayofMonth/Origin categoricals; a row's OOF encoding is (group sum - own-fold sum)/(...), so given the group, the encoding carries information about the row's own label (same mechanism as the leave-one-out TE leak). Groups are tiny (~2 rows), so the leak is strong.
Clean alternative: disjoint split — fit tables on one part of train, fit the model on the other.

## Exp 9 — 478a01d — disjoint-split target encoding (exploration) → keep
Tables (smoothed mean, m=20) fit on a stratified 50% of train; model fit on the other 50% (90/10 for early stopping). Eval 0.7525 (+0.0075) despite the model seeing only half the rows; val 0.7477 is now honest (slightly below eval).
Next: more date-level keys; later use both halves (swap roles, average).

## Exp 10 — 3559916 — more TE keys (date x origin x 4h block, date x route) → discard
Eval 0.7517 (-0.0008), eval time 65s. Finer keys are too sparse with 100K table rows.

## Exp 11 — 8f873dc — swapped-halves ensemble (follow-up) → keep
Model 0: tables half 0, fit on half 1; model 1 the reverse; average probabilities. prepare emits both halves' encodings, a small wrapper routes them. Eval 0.7609 (+0.0084). Uses all rows for both tables and fitting, plus some ensembling benefit. Eval 60s.
Next: K parts — model k fit on the other K-1 parts with tables from part k (more model data, sparser tables).

## Exp 12 — 429db26 — K=3 parts (follow-up) → keep
Eval 0.7640 (+0.0031). More model data beats denser tables; 3-model average. Train 19s, eval 73s.

## Exp 13 — 787e5b9 — K=5 parts → discard
Eval 0.7626 (-0.0014), train 42s (near the 60s limit), eval 100s. Tables from 20% of train are too sparse; K=3 is the sweet spot.

## Exp 14 — 17f0fcb — date-only TE key → discard
Eval 0.7633 (-0.0007). Date-level effect is already captured through the date x origin/dest/carrier encodings.

## Exp 15 — ba70753 — depth 10 (follow-up, tuning) → keep
Eval 0.7649 (+0.0009). Deeper trees help a little (interactions of TE features with airport/carrier/time).

## Exp 16 — 9ef21d9 — lr 0.03 → discard
Eval 0.7649 (equal), slower. lr not the bottleneck.

## Exp 17/18 — TE smoothing 5 / 50 → discard
Eval 0.7640 / 0.7647 vs 0.7649 at m=20. Flat; keep 20.

## Synthesis after 18 experiments
- Helps: early stopping + capacity; row/col subsampling; small max_cat_threshold (4) to stop overfitting on high-cardinality categoricals; date x origin/dest/carrier target encodings (weather/disruption days) fit on disjoint parts (K=3) with a 3-model average.
- Doesn't: OOF (hash-fold) TE — leaks through group identity; finer/sparser keys; lr, smoothing tweaks.
- Theory: the remaining signal is in high-cardinality interactions that max_cat_threshold=4 prevents the trees from carving out directly. TE of static interactions (route, carrier x origin, origin x hour...) may supply them as dense numeric features.

## Research (after exp 18)
- CatBoost (Prokhorenkova et al., https://arxiv.org/pdf/1706.09516): greedily builds combinations of categoricals and encodes them with target statistics; feature combinations give large gains. Adapting: add TE for static pairs (route, carrier x origin, origin/dest/carrier x hour) using the same leak-free disjoint-parts scheme.

## Exp 19 — 2a15c7e — TE of static pairs (exploration, CatBoost-inspired) → keep
Eval 0.7663 (+0.0014). But eval 168s: prepare rebuilds every key string once per part. Optimise next.

## Exp 20 — b078617 — build keys once per prepare call (simplification/speed) → keep
Identical AUC 0.7663, eval 99s (was 168s).

## Exp 21 — 9289413 — 4 more static TE keys (follow-up) → keep
Eval 0.7670 (+0.0007), eval 131s. Diminishing returns from more keys; eval time is now the constraint → optimise prepare.

## Exp 22 — df09b6b — faster prepare (simplification/speed) → keep
Identical AUC 0.7670; eval 39s (was 131s). Frees eval budget for more models.

## Exp 23 — 10cc0b6 — 6-model bag (3 parts x 2 split seeds) (follow-up) → keep
Eval 0.7683 (+0.0013). Training 54s — too close to the 60s limit; reduce training cost next.

## Exp 24 — early_stopping_rounds 50 → discard
Eval 0.7678 (-0.0005) though training 38s.

## Exp 25 — efbae97 (crash, stratify bug) / 09631d2 — prepare(train) once and slice per part (speed) → keep
Identical AUC 0.7683, training 49s (was 54s).

## Exp 26 — colsample 0.6 → discard (0.7680)

## Exp 27 — 9240ea5 — drop raw Origin/Dest from the models (simplification) → keep
Eval 0.7686 (+0.0003), training 32s (was 49s). The airport information is carried by the TE features; the raw ~300-level categoricals were mostly overfitting.

## Exp 28 — 09a2c76 — back to default max_cat_threshold (simplification) → keep
Eval 0.7691 (+0.0005). The threshold only mattered for the high-cardinality airports.

## Exp 29 — 334db8a — 9-model bag → discard (for now)
Eval 0.7695 (+0.0004) but training 51s of the 60s limit; not worth the timeout risk for later experiments. Revisit at the end if headroom allows.

## Exp 30 — 957a32a — date x hour TE → discard (0.7688)

## Synthesis after 30 experiments (best 0.7691)
- Wins since exp 18: static-pair TEs (CatBoost-style combinations), dropping raw Origin/Dest, default cat threshold, bagging over split seeds (limited by the 60s training cap), and big speedups of prepare/training.
- Flat/negative: more/finer date keys, lr, smoothing, colsample.
- Theory: date x airport signal is limited by sparsity (~0.7 rows per key per table). Weather/disruption persists across adjacent days → borrowing neighbouring days' rates at the same airport could densify the signal.

## Research (after exp 30)
- Delay literature on temporal persistence / lagged disruption effects (e.g. https://arxiv.org/pdf/2103.11221, https://www.ischool.berkeley.edu/projects/2025/air-travel-delay-prediction-feature-engineering-and-ml-approaches): disruptions linger. Adapted as neighbouring-day lookups in the date x airport tables.

## Exp 31 — 19a3fee — previous/next-day rates at same origin/dest (exploration) → keep
Date keys switched to day-of-year; extra features look up DateOrigin/DateDest tables at doy±1. Eval 0.7699 (+0.0008).

## Exp 32 — +-2 day lookups → discard (0.7701, +0.0002 within noise, +6s training)
## Exp 33 — b08bad8 — min_child_weight 5 → discard (equal 0.7699) but training 31s vs 38s: useful to buy headroom for more bagging.

## Exp 34 — 1dcb4bc — min_child_weight 5 + 9-model bag (follow-up) → keep
Eval 0.7704 (+0.0005). mcw 5 paid for the third split seed; training 47s.

## Exp 35 — previous/next-day carrier rates → discard (equal 0.7704)
## Exp 36 — 4f71d30 — hierarchical smoothing (date x X shrunk toward day rate) → discard (0.7685)
Shrinking toward the day's rate makes the date x airport encodings less specific; global prior works better here.

## Exp 37 — drop raw Month/DayofMonth → discard (0.7685)
## Exp 38 — reg_lambda 5 → discard (0.7705, noise, slower)
## Exp 39 — 17237b2 — max_bin 64 → keep (0.7706, +0.0002, training a bit faster)

## Exp 40 — depth 12 → discard (0.7704)

## Synthesis after 40 experiments (best 0.7706)
Hyperparameters saturated (lr, depth, lambda, colsample, smoothing all flat). Training time (60s cap) is the binding constraint for bagging. Real gains came from features: static-pair TEs and neighbouring-day airport rates. Next: pooled 3-day window rates (denser than separate day lookups).

## Exp 41 — pooled 3-day window rates → discard (equal 0.7706)
## Exp 42 — early-stopping val 5% → discard (0.7703)

## Research (plateau after exp 40–42)
- Kaggle playground 1st-place write-up (Deotte, https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi): diverse models + target encodings of combinations are the main levers.
- Gap in my encodings: the individual scheduled flight (carrier + route + exact CRSDepTime) — repeats daily, has its own punctuality history.

## Exp 43 — e9cac85 — scheduled-flight TE (exploration) → keep
Eval 0.7710 (+0.0004).

## Exp 44 — 30a45fa — carrier x origin x hour TE → keep (0.7712, +0.0002, one line)

## Exp 45 — origin x weekday x hour TE → discard (0.7707)
## Exp 46 — depth 8/10/12 by split seed (diversity) → discard (0.7710)
## Exp 47 — fcdd061 — numeric day-of-year feature → discard (0.7711)

## Exp 48 — log-odds averaging → discard (equal 0.7712)
## Exp 49 — predict with full-train tables → discard (0.7691): models trained on noisier 1/3-tables; denser tables at prediction are a distribution shift.
## Exp 50 — 3c9a647 — single origin/dest TE → keep (0.7716, +0.0004)
Since raw Origin/Dest were dropped (exp 27), no feature carried plain airport rates.

## Exp 51 — month x dest, dest x weekday → discard (equal)
## Exp 52 — 10d6952 — ablation: drop origin x weekday, month x origin → keep (equal 0.7716, simpler, training 44s)
## Exp 53 — 8c8b36b — ablation: drop carrier x hour, carrier x route → keep (0.7717, simpler)
## Exp 54 — ablation: drop route x hour → discard (0.7715)
## Exp 55 — lr 0.07 + 4 split seeds (12 models) → discard (equal 0.7717, slower)
## Exp 56 — TE smoothing 10 → discard (equal 0.7717)
## Exp 57 — subsample 0.7 → discard (0.7712)
## Exp 58 — ablation: drop dest x hour → discard (0.7711)
## Exp 59 — date x origin x carrier TE → discard (0.7716)

## Final summary
- **Best Eval AUC: 0.7717 at commit 8c8b36b** (baseline 0.7203 at fdbad98, +0.051). 59 experiments in ~1h40m of the 2h budget.
- **Final model:** a bag of 9 XGBoost models (3 disjoint parts x 3 split seeds). Model k is fit on 2/3 of train and gets target-encoded features from tables fit on the remaining 1/3, so no training row's features contain its own label. Depth 10, lr 0.05, subsample/colsample 0.8, min_child_weight 5, max_bin 64, early stopping on 10% of each fit part. The raw Origin/Dest categoricals are dropped in favour of the encodings. Training ~45s, eval ~45s.
- **Encodings (smoothed means, m=20):** date x origin/dest/carrier; origin/dest on the previous and next day; origin, dest, route, carrier x origin, origin/dest x hour, route x hour, scheduled flight (carrier+route+CRSDepTime), carrier x origin x hour.
- **What worked (roughly by size):** early stopping + capacity (+0.012); date x airport/carrier encodings on disjoint splits and the K-part bag (+0.020 together); subsampling and max_cat_threshold for the raw airport categoricals (+0.013, later made unnecessary by dropping them); static-pair CatBoost-style encodings (+0.002); neighbouring-day airport rates (+0.0008); bagging over split seeds (+0.001); flight and single-airport encodings (+0.0008).
- **What didn't:** out-of-fold (hash-fold) target encoding (leaks through group identity, and validation looked great while eval dropped); full-train tables at prediction time (distribution shift); finer or sparser date keys; hierarchical smoothing; lr, depth, lambda, colsample, smoothing and early-stopping tweaks once the features were in place.
- **Constraint:** the 60s training cap limits bagging. The 9-model bag sits at ~45s.
- **Next to try:** a faster per-model setup (e.g. hist with fewer rounds at a higher lr, tuned jointly) to afford more split seeds; pooling encodings from several table parts per model (e.g. tables from 2 of 3 other parts in a 4-part scheme); explicit weather-event features built from date x region clusters of airports learned from train.
