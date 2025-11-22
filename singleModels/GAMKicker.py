# GAM baseline for kicker probabilities (LogisticGAM via pygam)
# - Temporal holdout (latest season) as TEST
# - 5-fold CV on earlier seasons for hyperparam tuning using Brier score on P(miss)
# - Per-distance isotonic calibration fit on TRAIN OOF predictions only (P(miss), no test leakage)
# - Final report on TEST:
#     * Brier (primary, on P(make))
#     * AUC (on P(make))
#     * PR-AUC(miss) (on P(miss))
#     * ECE (on P(make))
#     * Confusion matrix at fixed threshold 0.5 on P(make)
#     * P(make)/P(miss) diagnostics by true outcome

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from pygam import LogisticGAM

RANDOM_STATE = 42
TARGET = "field_goal_result_binary"  # 1=make, 0=miss
CATEGORICAL = ["kicker_player_name"]
PARQUET = "field_goals_model_ready.parquet"
CSV = "field_goals_model_ready.csv"

N_ECE_BINS = 10
BINS       = (0, 40, 50, 80)  # distance bins for per-range isotonic


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
    # class-balancing sample weights; y: 1=miss, 0=make
    y = pd.Series(y)
    cnt = y.value_counts()
    tot = len(y)
    return y.map({c: tot / (2 * cnt[c]) for c in cnt.index}).values


def fold_target_encode(train_col, train_y, valid_col, smoothing=20.0):
    # train_y: 1=miss, 0=make → encode miss rate per kicker
    miss = (train_y == 1).astype(int)
    prior = miss.mean()
    g = (
        pd.DataFrame({"k": train_col.values, "miss": miss})
        .groupby("k")["miss"]
        .agg(["mean", "count"])
    )
    enc = (g["count"] * g["mean"] + smoothing * prior) / (g["count"] + smoothing)
    return valid_col.map(enc).fillna(prior).values


def fit_isotonic_by_range(x_dist, p, y, bins=BINS):
    """
    Fit per-distance-bin isotonic calibrators.
    x_dist: kick_distance
    p: predicted probabilities (here P(miss))
    y: true labels (1=miss, 0=make)
    """
    irs = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x_dist >= lo) & (x_dist < hi)
        if m.sum() > 50:  # need enough data per bin
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
    """
    Expected Calibration Error.
    probs: P(make), y: 1=make, 0=miss
    """
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
    """
    Returns:
        X: features
        y: target (1=miss, 0=make)
    """
    y_make = df[TARGET].astype(int).values  # 1=make, 0=miss
    y = (1 - y_make).astype(int)           # 1=miss, 0=make
    X = df.drop(columns=[TARGET]).copy()
    num = [c for c in X.columns if c not in CATEGORICAL]
    cat = CATEGORICAL[0]
    return X, y, num, cat


def add_te_and_scale(X_tr, X_va, y_tr, num_cols, cat):
    """
    Add kicker target encoding (miss rate), then standardize numeric+TE.
    """
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    # target encoding for kicker miss rate
    X_tr["kicker_te"] = fold_target_encode(X_tr[cat], pd.Series(y_tr), X_tr[cat])
    X_va["kicker_te"] = fold_target_encode(X_tr[cat], pd.Series(y_tr), X_va[cat])

    use_cols = num_cols + ["kicker_te"]

    scaler = StandardScaler()
    Xtr_mat = scaler.fit_transform(X_tr[use_cols])
    Xva_mat = scaler.transform(X_va[use_cols])
    return Xtr_mat, Xva_mat, use_cols, scaler


def safe_probs(p, fallback_len):
    p = np.asarray(p).ravel()
    if p.shape[0] != fallback_len or not np.all(np.isfinite(p)):
        # fallback: neutral 0.5 for all examples
        return np.full(fallback_len, 0.5, dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)


def main():
    df = load_df()
    latest_season = int(df["season"].max())
    test = df[df["season"] == latest_season].reset_index(drop=True)
    train = df[df["season"] < latest_season].reset_index(drop=True)
    print(f"Train seasons ≤ {latest_season-1}: {len(train)} rows | Test season {latest_season}: {len(test)} rows")

    X_tr_all, y_tr_all, num, cat = build_xy(train)
    X_te_raw, y_te_miss, _, _ = build_xy(test)  # y_te_miss: 1=miss, 0=make
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Random search over lam and n_splines
    gam_param_space = {
        "lam":       [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "n_splines": [8, 10, 12, 15, 20],
    }

    def cv_score_gam(params, idx=None, total=None):
        lam = params["lam"]
        n_splines = params["n_splines"]
        briers = []
        for fold_id, (tr_idx, va_idx) in enumerate(
            skf.split(X_tr_all, y_tr_all), start=1
        ):
            Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
            ytr, yva = y_tr_all[tr_idx], y_tr_all[va_idx]  # 1=miss, 0=make
            Xtr_mat, Xva_mat, use_cols, _ = add_te_and_scale(Xtr, Xva, ytr, num, cat)
            sw = compute_sw(ytr)

            gam = LogisticGAM(
                lam=lam,
                n_splines=n_splines,
                max_iter=400,
                fit_intercept=True
            )
            try:
                gam.fit(Xtr_mat, ytr, weights=sw)
                p_miss = gam.predict_proba(Xva_mat)  # P(y=1=miss)
            except Exception as e:
                print(
                    f"[GAM CV] cand {idx}/{total} lam={lam} n_splines={n_splines} "
                    f"fold {fold_id} crashed with error: {e}"
                )
                return 1.0

            p_miss = safe_probs(p_miss, len(yva))
            fold_brier = brier_score_loss(yva, p_miss)
            briers.append(fold_brier)
            if idx is not None and total is not None:
                print(
                    f"  [GAM CV] cand {idx}/{total}, fold {fold_id}/5, "
                    f"Brier(P(miss))={fold_brier:.5f}",
                    flush=True,
                )
        return float(np.mean(briers))

    rng = np.random.RandomState(RANDOM_STATE)
    N_CANDIDATES = 40
    candidates = [
        {k: rng.choice(v) for k, v in gam_param_space.items()}
        for _ in range(N_CANDIDATES)
    ]

    best_params = None
    best_score = np.inf
    for i, c in enumerate(candidates, 1):
        s = cv_score_gam(c, idx=i, total=N_CANDIDATES)
        if s < best_score:
            best_score, best_params = s, c
        print(
            f"[GAM Tune] {i:02d}/{N_CANDIDATES} lam={c['lam']:.4g} "
            f"n_splines={c['n_splines']} Brier(P(miss))={s:.5f} "
            f"best={best_score:.5f}",
            flush=True,
        )

    print(f"\nBest GAM params (by Brier on P(miss)): {best_params}")

    best_lam = best_params["lam"]
    best_n_splines = best_params["n_splines"]

    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all.copy()  # 1=miss, 0=make
    oof_dist   = train["kick_distance"].values

    for fold_id, (tr_idx, va_idx) in enumerate(
        skf.split(X_tr_all, oof_y_miss), start=1
    ):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]
        Xtr_mat, Xva_mat, use_cols, _ = add_te_and_scale(Xtr, Xva, ytr, num, cat)
        sw = compute_sw(ytr)

        gam = LogisticGAM(
            lam=best_lam,
            n_splines=best_n_splines,
            max_iter=400,
            fit_intercept=True
        )
        try:
            gam.fit(Xtr_mat, ytr, weights=sw)
            p_fold = gam.predict_proba(Xva_mat)
        except Exception as e:
            print(
                f"[GAM OOF] lam={best_lam} n_splines={best_n_splines} "
                f"fold {fold_id} crashed with error: {e}"
            )
            p_fold = np.full(len(yva), 0.5, dtype=float)

        p_fold = safe_probs(p_fold, len(yva))
        oof_p_miss[va_idx] = p_fold

    # calibrate P(miss) by distance
    irs_global = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)
    oof_p_miss_cal = apply_isotonic_by_range(oof_dist, oof_p_miss, irs_global, bins=BINS)
    # OOF Brier on P(miss)
    oof_brier_miss = brier_score_loss(oof_y_miss, oof_p_miss_cal)
    print(f"OOF Brier after per-distance isotonic calibration (P(miss)): {oof_brier_miss:.5f}")

    # Fit scaler/TE on all train, apply to test
    Xtr_mat_all, _, use_cols, scaler = add_te_and_scale(
        X_tr_all, X_tr_all, y_tr_all, num, cat
    )

    test_enc = test.copy()
    test_enc["kicker_te"] = fold_target_encode(
        train[CATEGORICAL[0]], pd.Series(y_tr_all), test[CATEGORICAL[0]]
    )
    Xte_mat = scaler.transform(test_enc[use_cols])

    gam_final = LogisticGAM(
        lam=best_lam,
        n_splines=best_n_splines,
        max_iter=400,
        fit_intercept=True
    )
    try:
        gam_final.fit(Xtr_mat_all, y_tr_all, weights=compute_sw(y_tr_all))
        pte_miss_raw = gam_final.predict_proba(Xte_mat)
    except Exception as e:
        print(
            f"[GAM FINAL] lam={best_lam} n_splines={best_n_splines} crashed with error: {e}, "
            "using 0.5 for all test probs"
        )
        pte_miss_raw = np.full(len(y_te_miss), 0.5, dtype=float)

    pte_miss_raw = safe_probs(pte_miss_raw, len(y_te_miss))
    pte_miss     = apply_isotonic_by_range(
        test["kick_distance"].values, pte_miss_raw, irs_global, bins=BINS
    )
    # derive P(make)
    y_te_make = 1 - y_te_miss          # 1=make, 0=miss
    pte_make  = 1.0 - pte_miss

    p_make_miss = pte_make[y_te_make == 0]  # predicted P(make) for true misses
    p_make_make = pte_make[y_te_make == 1]  # predicted P(make) for true makes

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

    print("\n=== Probability Diagnostics on TEST (GAM) ===")
    summarize("True MISSES (y=0) – P(make)", p_make_miss)
    summarize("True MAKES (y=1) – P(make)", p_make_make)

    for t in [0.60, 0.70, 0.80, 0.90]:
        frac = np.mean(p_make_miss < t)
        print(f"Fraction of MISSES with P(make) < {t:.2f}: {frac:.3f}")

    # Brier on P(make) (equivalent to Brier on P(miss))
    brier = brier_score_loss(y_te_make, pte_make)
    # AUC on P(make) vs y_make
    auc = roc_auc_score(y_te_make, pte_make)
    # PR-AUC for misses: use y_miss and P(miss)
    pr_miss = average_precision_score(y_te_miss, pte_miss)
    # ECE on P(make)
    ece10 = ece_score(pte_make, y_te_make, n_bins=N_ECE_BINS)

    # fixed threshold at 0.5 on P(make) for confusion matrix
    thr = 0.5
    yhat_make = (pte_make >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te_make, yhat_make).ravel()

    print("\n=== TEST (latest season) — GAM (P(make)/P(miss)) ===")
    print(f"Brier={brier:.5f} (primary, on P(make))")
    print(f"AUC={auc:.4f} | PR-AUC(miss)={pr_miss:.4f} | ECE@{N_ECE_BINS}={ece10:.4f}")
    print(f"Threshold for confusion matrix (on P(make)): {thr:.2f}")
    print(f"Confusion matrix (1=make, 0=miss): tn={tn} fp={fp} fn={fn} tp={tp}")


if __name__ == "__main__":
    main()
