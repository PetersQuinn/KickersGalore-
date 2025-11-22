# "Bayesian" (bootstrap-ensemble) logistic regression baseline for kicker probabilities
# - Temporal holdout (latest season) as TEST
# - 5-fold CV on earlier seasons for hyperparam tuning using Brier score on P(make)
# - Each model in CV and final fit is an ensemble of bootstrapped LogisticRegression models
# - approximates posterior predictive averaging over coefficient uncertainty
# - Per-distance isotonic calibration fit on TRAIN OOF predictions only (no test leakage)
# - Fixed threshold 0.5 on calibrated P(make) for confusion matrix
# - Final report on TEST:
#     * Brier (primary, on P(make))
#     * AUC (on P(make))
#     * PR-AUC(miss) (on P(miss))
#     * ECE (on P(make))
#     * Confusion matrix
#     * P(make) diagnostics for true makes vs misses

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
TARGET = "field_goal_result_binary"  # 1=make, 0=miss
CATEGORICAL = ["kicker_player_name"]
PARQUET = "field_goals_model_ready.parquet"
CSV = "field_goals_model_ready.csv"

N_ECE_BINS = 10
BINS = (0, 40, 50, 80)   
N_BOOTSTRAP = 10


def load_df():
    if Path(PARQUET).exists():
        try:
            print(f"Loading PARQUET: {PARQUET}")
            return pd.read_parquet(PARQUET)
        except Exception as e:
            print(f"[WARN] Failed to read {PARQUET} ({e}). Falling back to CSV.")
    print(f"Loading CSV: {CSV}")
    return pd.read_csv(CSV)


def compute_sw(y):
    y = pd.Series(y)
    cnt = y.value_counts()
    tot = len(y)
    return y.map({c: tot / (2 * cnt[c]) for c in cnt.index}).values


def fold_target_encode(train_col, train_y, valid_col, smoothing=20.0):
    miss = (train_y == 0).astype(int)
    prior = miss.mean()
    g = (
        pd.DataFrame({"k": train_col.values, "miss": miss})
        .groupby("k")["miss"]
        .agg(["mean", "count"])
    )
    enc = (g["count"] * g["mean"] + smoothing * prior) / (g["count"] + smoothing)
    return valid_col.map(enc).fillna(prior).values


def fit_isotonic_by_range(x_dist, p, y, bins=BINS):
    # p: P(make), y: 1=make, 0=miss
    irs = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x_dist >= lo) & (x_dist < hi)
        if m.sum() > 50:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p[m], y[m])
            irs[(lo, hi)] = ir
    return irs


def apply_isotonic_by_range(x_dist, p, irs, bins=BINS):
    p_cal = np.array(p, dtype=float, copy=True)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x_dist >= lo) & (x_dist < hi)
        ir = irs.get((lo, hi))
        if ir is not None and m.any():
            p_cal[m] = ir.transform(p[m])
    return p_cal


def ece_score(probs, y, n_bins=N_ECE_BINS):
    # probs: P(make), y: 1=make, 0=miss
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        e += m.sum() / N * abs(probs[m].mean() - y[m].mean())
    return float(e)


def build_xy(df):
    y_make = df[TARGET].astype(int).values  # 1=make, 0=miss
    y = y_make
    X = df.drop(columns=[TARGET]).copy()
    num = [c for c in X.columns if c not in CATEGORICAL]
    cat = CATEGORICAL[0]
    return X, y, num, cat


def add_te_and_scale(X_tr, X_va, y_tr, num_cols, cat):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    # add target encoding for kicker (miss rate)
    X_tr["kicker_te"] = fold_target_encode(X_tr[cat], pd.Series(y_tr), X_tr[cat])
    X_va["kicker_te"] = fold_target_encode(X_tr[cat], pd.Series(y_tr), X_va[cat])
    use_cols = num_cols + ["kicker_te"]
    # scale numeric + TE
    scaler = StandardScaler()
    Xtr_mat = scaler.fit_transform(X_tr[use_cols])
    Xva_mat = scaler.transform(X_va[use_cols])
    return Xtr_mat, Xva_mat, use_cols, scaler


def fit_bootstrap_logreg(
    X,
    y,
    C,
    penalty="l2",
    sw=None,
    n_bootstrap=N_BOOTSTRAP,
    random_state=RANDOM_STATE,
):
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    if sw is not None:
        sw = np.asarray(sw)
    n = len(y)
    models = []
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        swb = sw[idx] if sw is not None else None
        clf = LogisticRegression(
            solver="liblinear",  # supports l1 and l2
            penalty=penalty,
            C=C,
            class_weight="balanced",
            random_state=rng.randint(0, 1_000_000),
        )
        clf.fit(Xb, yb, sample_weight=swb)
        models.append(clf)
    return models


def predict_bootstrap(models, X):
    """
    Average P(make) across all bootstrap models.
    """
    X = np.asarray(X)
    probs = [m.predict_proba(X)[:, 1] for m in models]  # P(make)
    return np.mean(probs, axis=0)



def main():
    df = load_df()
    latest_season = int(df["season"].max())
    test = df[df["season"] == latest_season].reset_index(drop=True)
    train = df[df["season"] < latest_season].reset_index(drop=True)
    print(
        f"Train seasons ≤ {latest_season-1}: {len(train)} rows | "
        f"Test season {latest_season}: {len(test)} rows"
    )

    X_tr_all, y_tr_all, num, cat = build_xy(train)      # y: 1=make, 0=miss
    X_te_raw, y_te, _, _ = build_xy(test)               # y_te: 1=make, 0=miss
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    C_grid = np.logspace(-3, 1, 30)  # 0.001 → 10 in log steps
    penalties = ["l1", "l2"]

    rng = np.random.RandomState(RANDOM_STATE)
    N_CANDIDATES = 50
    candidates = [
        {
            "C": float(rng.choice(C_grid)),
            "penalty": str(rng.choice(penalties)),
        }
        for _ in range(N_CANDIDATES)
    ]

    best_params = None
    best_brier = np.inf

    def cv_score_bayes(params, idx=None, total=None):
        """
        Return mean CV Brier on P(make) from the bootstrap ensemble
        for a given (C, penalty) candidate.
        """
        briers = []
        for fold_id, (tr_idx, va_idx) in enumerate(
            skf.split(X_tr_all, y_tr_all), start=1
        ):
            Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
            ytr, yva = y_tr_all[tr_idx], y_tr_all[va_idx]
            Xtr_mat, Xva_mat, use_cols, _ = add_te_and_scale(Xtr, Xva, ytr, num, cat)
            sw = compute_sw(ytr)

            models = fit_bootstrap_logreg(
                Xtr_mat,
                ytr,
                C=params["C"],
                penalty=params["penalty"],
                sw=sw,
                n_bootstrap=N_BOOTSTRAP,
                random_state=RANDOM_STATE + idx * 100 + fold_id,
            )
            p_make = predict_bootstrap(models, Xva_mat)
            fold_brier = brier_score_loss(yva, p_make)
            briers.append(fold_brier)
            if idx is not None and total is not None:
                print(
                    f"  [Bayes-LR CV] cand {idx}/{total}, "
                    f"fold {fold_id}/5, Brier(P(make))={fold_brier:.5f}",
                    flush=True,
                )
        return float(np.mean(briers))

    for i, params in enumerate(candidates, start=1):
        s = cv_score_bayes(params, idx=i, total=N_CANDIDATES)
        if s < best_brier:
            best_brier = s
            best_params = params
        print(
            f"[Bayes-LR Tune] {i:02d}/{N_CANDIDATES} "
            f"Brier(P(make))={s:.5f} best={best_brier:.5f} "
            f"params={params}",
            flush=True,
        )

    print("\nBest Bayes-LR params (by Brier on P(make)):", best_params)

    oof_p = np.zeros(len(train))      # raw P(make) from bootstrap ensemble
    oof_y = y_tr_all.copy()           # 1=make, 0=miss
    oof_dist = train["kick_distance"].values

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_tr_all, oof_y), start=1):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y[tr_idx], oof_y[va_idx]
        Xtr_mat, Xva_mat, use_cols, _ = add_te_and_scale(Xtr, Xva, ytr, num, cat)
        sw = compute_sw(ytr)

        models = fit_bootstrap_logreg(
            Xtr_mat,
            ytr,
            C=best_params["C"],
            penalty=best_params["penalty"],
            sw=sw,
            n_bootstrap=N_BOOTSTRAP,
            random_state=RANDOM_STATE + 1000 + fold_id,
        )
        oof_p[va_idx] = predict_bootstrap(models, Xva_mat)  

    irs_global = fit_isotonic_by_range(oof_dist, oof_p, oof_y, bins=BINS)
    oof_p_cal = apply_isotonic_by_range(oof_dist, oof_p, irs_global, bins=BINS)
    oof_brier = brier_score_loss(oof_y, oof_p_cal)
    print(
        f"OOF Brier after per-distance isotonic calibration (P(make)): "
        f"{oof_brier:.5f}"
    )

    Xtr_mat_all, _, use_cols, scaler = add_te_and_scale(
        X_tr_all, X_tr_all, y_tr_all, num, cat
    )
    test_enc = test.copy()
    test_enc["kicker_te"] = fold_target_encode(
        train[cat], pd.Series(y_tr_all), test[cat]
    )
    Xte_mat = scaler.transform(test_enc[use_cols])

    # final bootstrap ensemble on ALL train
    sw_all = compute_sw(y_tr_all)
    final_models = fit_bootstrap_logreg(
        Xtr_mat_all,
        y_tr_all,
        C=best_params["C"],
        penalty=best_params["penalty"],
        sw=sw_all,
        n_bootstrap=N_BOOTSTRAP,
        random_state=RANDOM_STATE + 9999,
    )

    pte_raw = predict_bootstrap(final_models, Xte_mat)  # raw P(make)
    pte = apply_isotonic_by_range(
        test["kick_distance"].values, pte_raw, irs_global, bins=BINS
    )  # calibrated P(make)

    p_make_miss = pte[y_te == 0]  # predicted P(make) for true misses
    p_make_make = pte[y_te == 1]  # predicted P(make) for true makes

    def summarize(name, arr):
        print(f"\n{name}:")
        print(f"  count = {len(arr)}")
        print(f"  mean  = {np.mean(arr):.4f}")
        print(f"  std   = {np.std(arr):.4f}")
        print(f"  min   = {np.min(arr):.4f}")
        print(f"  10%   = {np.percentile(arr, 10):.4f}")
        print(f"  25%   = {np.percentile(arr, 25):.4f}")
        print(f"  50%   = {np.percentile(arr, 50):.4f}")
        print(f"  75%   = {np.percentile(arr, 75):.4f}")
        print(f"  90%   = {np.percentile(arr, 90):.4f}")
        print(f"  max   = {np.max(arr):.4f}")

    print("\n=== Probability Diagnostics on TEST (Bayes-LR) ===")
    summarize("True MISSES (y=0) – P(make)", p_make_miss)
    summarize("True MAKES (y=1) – P(make)", p_make_make)

    for t in [0.60, 0.70, 0.80, 0.90]:
        frac = np.mean(p_make_miss < t)
        print(f"Fraction of MISSES with P(make) < {t:.2f}: {frac:.3f}")

    thr = 0.5
    yhat = (pte >= thr).astype(int)

    # Brier on P(make)
    brier = brier_score_loss(y_te, pte)
    # AUC on P(make) vs y (1=make)
    auc = roc_auc_score(y_te, pte)
    # PR-AUC for misses: flip labels and probs
    y_miss = 1 - y_te
    p_miss = 1 - pte
    pr_miss = average_precision_score(y_miss, p_miss)
    ece10 = ece_score(pte, y_te, n_bins=N_ECE_BINS)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()

    print("\n=== TEST (latest season) — Bayesian-style Logistic Regression (P(make)) ===")
    print(f"Brier={brier:.5f} (primary, on P(make))")
    print(f"AUC={auc:.4f} | PR-AUC(miss)={pr_miss:.4f} | ECE@{N_ECE_BINS}={ece10:.4f}")
    print(f"Threshold for confusion matrix (on P(make)): {thr:.2f}")
    print(f"Confusion matrix (1=make, 0=miss): tn={tn} fp={fp} fn={fn} tp={tp}")


if __name__ == "__main__":
    main()
