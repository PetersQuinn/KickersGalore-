# kicker_prob_hist_all_models.py
#
# One-stop script:
# - Load field_goals_model_ready.(parquet/csv)
# - Hold out latest season as TEST
# - For each tuned model:
#       * Fit model with fixed hyperparameters
#       * Fit per-distance isotonic calibration on TRAIN (OOF)
#       * Get calibrated P(make) on TEST
#       * Plot histograms of P(make) for true makes vs misses
#
# Models:
#   - Bagging (DecisionTree base)
#   - Bayesian-style Logistic Regression (bootstrap LR)
#   - LightGBM
#   - GAM (pygam LogisticGAM)
#   - BART (gbart)
#   - Plain Logistic Regression
#
# Output:
#   prob_histograms/<model_name>_hist.png

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from pygam import LogisticGAM
from gbart.modified_bartpy.sklearnmodel import SklearnModel

# ----------------- CONFIG -----------------

RANDOM_STATE = 42
TARGET = "field_goal_result_binary"  # 1=make, 0=miss
CATEGORICAL = ["kicker_player_name"]
PARQUET = "field_goals_model_ready.parquet"
CSV = "field_goals_model_ready.csv"

N_ECE_BINS = 10
BINS = (0, 40, 50, 80)  # distance bins for per-range isotonic
N_FOLDS = 5

# Tuned hyperparameters you provided
BAGGING_BEST = {
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_samples": 0.9,
    "max_features": 0.8,
}

BAYES_LR_C = 0.001
BAYES_LR_BOOTSTRAP = 10

LGBM_BEST = {
    "num_leaves": 127,
    "learning_rate": 0.015,
    "min_child_samples": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.05,
    "reg_lambda": 0.1,
    "n_estimators": 1800,
    "max_depth": -1,
    "min_split_gain": 0.0,
}

GAM_BEST = {
    "lam": 1000.0,
    "n_splines": 8,
}

BART_BEST = {
    "n_trees": 200,
    "n_samples": 150,
    "n_burn": 400,
    "thin": 0.5,
}

LR_BEST_C = 0.00126743

OUT_DIR = "prob_histograms"


# ----------------- COMMON HELPERS -----------------

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
    """
    Class-balancing sample weights.
    y can be 0/1 with either convention; weights just equalize classes.
    """
    y = pd.Series(y)
    cnt = y.value_counts()
    tot = len(y)
    return y.map({c: tot / (2 * cnt[c]) for c in cnt.index}).values


def fold_target_encode_miss(train_col, train_y_miss, valid_col, smoothing=20.0):
    """
    Target encoding using miss rate per kicker.
    train_y_miss: 1=miss, 0=make.
    """
    miss = (train_y_miss == 1).astype(int)
    prior = miss.mean()
    g = (
        pd.DataFrame({"k": train_col.values, "miss": miss})
        .groupby("k")["miss"]
        .agg(["mean", "count"])
    )
    enc = (g["count"] * g["mean"] + smoothing * prior) / (g["count"] + smoothing)
    return valid_col.map(enc).fillna(prior).values


def fold_target_encode_from_make(train_col, train_y_make, valid_col, smoothing=20.0):
    """
    Used for LR/Bayes-LR where y is 1=make, 0=miss.
    We still encode miss rate per kicker.
    """
    miss = (train_y_make == 0).astype(int)
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
    p: predicted probabilities (P(event) with y=1)
    y: 0/1 labels matching p
    """
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


def safe_probs(p, fallback_len):
    """
    Ensure probabilities are finite and in (0,1). If not, fall back to 0.5.
    """
    p = np.asarray(p).ravel()
    if p.shape[0] != fallback_len or not np.all(np.isfinite(p)):
        return np.full(fallback_len, 0.5, dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)


def plot_hist(model_name, p_make, y_true, out_dir=OUT_DIR):
    """
    Histogram of P(make) for true misses vs makes.
    X-axis: 0–1 in 0.01 bins
    Y-axis: frequency
    """
    os.makedirs(out_dir, exist_ok=True)
    bins = np.linspace(0.0, 1.0, 101)  # 0.01-wide

    p_make = np.asarray(p_make)
    y_true = np.asarray(y_true)

    p_miss = p_make[y_true == 0]
    p_make_only = p_make[y_true == 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        p_miss,
        bins=bins,
        alpha=0.6,
        label="True MISSES",
        edgecolor="black",
    )
    ax.hist(
        p_make_only,
        bins=bins,
        alpha=0.6,
        label="True MAKES",
        edgecolor="black",
    )

    ax.set_xlabel("Predicted probability of MAKE")
    ax.set_ylabel("Frequency (number of kicks)")
    ax.set_title(f"{model_name}: P(make) distribution (makes vs misses)")
    ax.set_xlim(0.0, 1.0)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{model_name.lower().replace(' ', '_')}_hist.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT] Saved histogram for {model_name}: {out_path}")


# ----------------- MODEL-SPECIFIC RUNNERS -----------------

def build_xy_miss(df):
    """
    y: 1=miss, 0=make; used for Bagging / LGBM / GAM / BART.
    """
    y_make = df[TARGET].astype(int).values  # 1=make, 0=miss
    y = (1 - y_make).astype(int)           # 1=miss, 0=make
    X = df.drop(columns=[TARGET]).copy()
    num = [c for c in X.columns if c not in CATEGORICAL]
    cat = CATEGORICAL[0]
    return X, y, num, cat


def build_xy_make(df):
    """
    y: 1=make, 0=miss; used for LR / Bayes-LR.
    """
    y = df[TARGET].astype(int).values  # 1=make, 0=miss
    X = df.drop(columns=[TARGET]).copy()
    num = [c for c in X.columns if c not in CATEGORICAL]
    cat = CATEGORICAL[0]
    return X, y, num, cat


def run_bagging(train, test):
    print("\n[Bagging] Fitting...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # OOF predictions for isotonic (on P(miss))
    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all.copy()
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        # target encoding miss rate
        Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xva[cat])
        # ordinal id
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(Xtr[[cat]])
        Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
        Xva["kicker_id"] = enc.transform(Xva[[cat]]).astype("int64")
        use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

        sw = compute_sw(ytr)

        base = DecisionTreeClassifier(
            max_depth=BAGGING_BEST["max_depth"],
            min_samples_leaf=BAGGING_BEST["min_samples_leaf"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model = BaggingClassifier(
            estimator=base,
            n_estimators=BAGGING_BEST["n_estimators"],
            max_samples=BAGGING_BEST["max_samples"],
            max_features=BAGGING_BEST["max_features"],
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        model.fit(Xtr[use_cols], ytr, sample_weight=sw)
        oof_p_miss[va_idx] = model.predict_proba(Xva[use_cols])[:, 1]

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    # Fit on all train, encode test, predict
    Xtr = X_tr_all.copy()
    Xte = test.copy()

    Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], y_tr_all, Xtr[cat])
    Xte["kicker_te"] = fold_target_encode_miss(train[cat], y_tr_all, Xte[cat])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xtr[[cat]])
    Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
    Xte["kicker_id"] = enc.transform(Xte[[cat]]).astype("int64")
    use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

    sw_all = compute_sw(y_tr_all)

    base_final = DecisionTreeClassifier(
        max_depth=BAGGING_BEST["max_depth"],
        min_samples_leaf=BAGGING_BEST["min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    final = BaggingClassifier(
        estimator=base_final,
        n_estimators=BAGGING_BEST["n_estimators"],
        max_samples=BAGGING_BEST["max_samples"],
        max_features=BAGGING_BEST["max_features"],
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final.fit(Xtr[use_cols], y_tr_all, sample_weight=sw_all)

    pte_miss_raw = final.predict_proba(Xte[use_cols])[:, 1]
    pte_miss = apply_isotonic_by_range(test["kick_distance"].values,
                                       pte_miss_raw, irs, bins=BINS)
    pte_miss = np.clip(pte_miss, 1e-6, 1 - 1e-6)

    y_te_make = 1 - y_te_miss
    pte_make = 1.0 - pte_miss
    return pte_make, y_te_make


def fit_bootstrap_logreg(X, y, C, sw=None,
                         n_bootstrap=BAYES_LR_BOOTSTRAP,
                         random_state=RANDOM_STATE):
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
            solver="liblinear",
            penalty="l2",
            C=C,
            class_weight="balanced",
            random_state=rng.randint(0, 1_000_000),
        )
        clf.fit(Xb, yb, sample_weight=swb)
        models.append(clf)
    return models


def predict_bootstrap(models, X):
    X = np.asarray(X)
    probs = [m.predict_proba(X)[:, 1] for m in models]  # P(make)
    return np.mean(probs, axis=0)


def run_bayes_lr(train, test):
    print("\n[Bayes-LR] Fitting...")
    X_tr_all, y_tr_all, num, cat = build_xy_make(train)
    X_te_raw, y_te, _, _ = build_xy_make(test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # OOF on P(make)
    oof_p = np.zeros(len(train))
    oof_y = y_tr_all.copy()
    oof_dist = train["kick_distance"].values

    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_tr_all, oof_y), start=1):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y[tr_idx], oof_y[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["kicker_te"] = fold_target_encode_from_make(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_target_encode_from_make(Xtr[cat], ytr, Xva[cat])
        use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

        scaler = StandardScaler()
        Xtr_mat = scaler.fit_transform(Xtr[use_cols])
        Xva_mat = scaler.transform(Xva[use_cols])

        sw = compute_sw(ytr)
        models = fit_bootstrap_logreg(
            Xtr_mat, ytr, C=BAYES_LR_C, sw=sw,
            n_bootstrap=BAYES_LR_BOOTSTRAP,
            random_state=RANDOM_STATE + fold_id * 1000,
        )
        p_va = predict_bootstrap(models, Xva_mat)
        p_va = safe_probs(p_va, len(yva))
        oof_p[va_idx] = p_va

    irs = fit_isotonic_by_range(oof_dist, oof_p, oof_y, bins=BINS)

    # Final fit on all train, apply to test
    Xtr = X_tr_all.copy()
    Xte = test.copy()

    Xtr["kicker_te"] = fold_target_encode_from_make(Xtr[cat], y_tr_all, Xtr[cat])
    Xte["kicker_te"] = fold_target_encode_from_make(train[cat], y_tr_all, Xte[cat])
    use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

    scaler = StandardScaler()
    Xtr_mat_all = scaler.fit_transform(Xtr[use_cols])
    Xte_mat = scaler.transform(Xte[use_cols])

    sw_all = compute_sw(y_tr_all)
    final_models = fit_bootstrap_logreg(
        Xtr_mat_all, y_tr_all, C=BAYES_LR_C, sw=sw_all,
        n_bootstrap=BAYES_LR_BOOTSTRAP,
        random_state=RANDOM_STATE + 9999,
    )
    pte_raw = predict_bootstrap(final_models, Xte_mat)
    pte_raw = safe_probs(pte_raw, len(y_te))
    pte = apply_isotonic_by_range(test["kick_distance"].values, pte_raw, irs, bins=BINS)
    pte = np.clip(pte, 1e-6, 1 - 1e-6)

    return pte, y_te  # already P(make)


def run_lgbm(train, test):
    print("\n[LightGBM] Fitting...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all.copy()
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xva[cat])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(Xtr[[cat]])
        Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
        Xva["kicker_id"] = enc.transform(Xva[[cat]]).astype("int64")
        use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

        sw = compute_sw(ytr)
        pos = ytr.sum()
        neg = (ytr == 0).sum()
        spw = max(1.0, neg / max(1, pos))

        mono = [0] * len(use_cols)
        if "kick_distance" in use_cols:
            mono[use_cols.index("kick_distance")] = 1

        model = lgb.LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=spw,
            monotone_constraints=mono,
            verbosity=-1,
            **LGBM_BEST,
        )
        model.fit(
            Xtr[use_cols],
            ytr,
            sample_weight=sw,
            eval_set=[(Xva[use_cols], yva)],
            eval_metric="auc",
            callbacks=[lgb.log_evaluation(0)],
        )
        oof_p_miss[va_idx] = model.predict_proba(Xva[use_cols])[:, 1]

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    Xtr = X_tr_all.copy()
    Xte = test.copy()

    Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], y_tr_all, Xtr[cat])
    Xte["kicker_te"] = fold_target_encode_miss(train[cat], y_tr_all, Xte[cat])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xtr[[cat]])
    Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
    Xte["kicker_id"] = enc.transform(Xte[[cat]]).astype("int64")
    use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

    pos = y_tr_all.sum()
    neg = (y_tr_all == 0).sum()
    spw = max(1.0, neg / max(1, pos))
    mono = [0] * len(use_cols)
    if "kick_distance" in use_cols:
        mono[use_cols.index("kick_distance")] = 1

    final = lgb.LGBMClassifier(
        objective="binary",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=spw,
        monotone_constraints=mono,
        verbosity=-1,
        **LGBM_BEST,
    )
    final.fit(
        Xtr[use_cols],
        y_tr_all,
        sample_weight=compute_sw(y_tr_all),
        callbacks=[lgb.log_evaluation(0)],
    )

    pte_miss_raw = final.predict_proba(Xte[use_cols])[:, 1]
    pte_miss = apply_isotonic_by_range(test["kick_distance"].values,
                                       pte_miss_raw, irs, bins=BINS)
    pte_miss = np.clip(pte_miss, 1e-6, 1 - 1e-6)
    y_te_make = 1 - y_te_miss
    pte_make = 1.0 - pte_miss
    return pte_make, y_te_make


def run_gam(train, test):
    print("\n[GAM] Fitting...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all.copy()
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xva[cat])
        use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

        scaler = StandardScaler()
        Xtr_mat = scaler.fit_transform(Xtr[use_cols])
        Xva_mat = scaler.transform(Xva[use_cols])

        sw = compute_sw(ytr)
        gam = LogisticGAM(
            lam=GAM_BEST["lam"],
            n_splines=GAM_BEST["n_splines"],
            max_iter=300,
            fit_intercept=True,
        )
        try:
            gam.fit(Xtr_mat, ytr, weights=sw)
            p_fold = gam.predict_proba(Xva_mat)
        except Exception as e:
            print(f"[GAM] fold crash: {e}")
            p_fold = np.full(len(yva), 0.5, dtype=float)

        p_fold = safe_probs(p_fold, len(yva))
        oof_p_miss[va_idx] = p_fold

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    Xtr = X_tr_all.copy()
    Xte = test.copy()
    Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], y_tr_all, Xtr[cat])
    Xte["kicker_te"] = fold_target_encode_miss(train[cat], y_tr_all, Xte[cat])
    use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

    scaler = StandardScaler()
    Xtr_mat_all = scaler.fit_transform(Xtr[use_cols])
    Xte_mat = scaler.transform(Xte[use_cols])

    gam_final = LogisticGAM(
        lam=GAM_BEST["lam"],
        n_splines=GAM_BEST["n_splines"],
        max_iter=300,
        fit_intercept=True,
    )
    try:
        gam_final.fit(Xtr_mat_all, y_tr_all, weights=compute_sw(y_tr_all))
        pte_miss_raw = gam_final.predict_proba(Xte_mat)
    except Exception as e:
        print(f"[GAM FINAL] crashed: {e}")
        pte_miss_raw = np.full(len(y_te_miss), 0.5, dtype=float)

    pte_miss_raw = safe_probs(pte_miss_raw, len(y_te_miss))
    pte_miss = apply_isotonic_by_range(test["kick_distance"].values,
                                       pte_miss_raw, irs, bins=BINS)
    pte_miss = np.clip(pte_miss, 1e-6, 1 - 1e-6)

    y_te_make = 1 - y_te_miss
    pte_make = 1.0 - pte_miss
    return pte_make, y_te_make


def run_bart(train, test):
    print("\n[BART] Fitting (this one may be slow)...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all.copy()
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_target_encode_miss(Xtr[cat], ytr, Xva[cat])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(Xtr[[cat]])
        Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
        Xva["kicker_id"] = enc.transform(Xva[[cat]]).astype("int64")
        use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

        m = SklearnModel(
            sublist=None,
            n_trees=int(BART_BEST["n_trees"]),
            n_chains=1,
            n_samples=int(BART_BEST["n_samples"]),
            n_burn=int(BART_BEST["n_burn"]),
            thin=float(BART_BEST["thin"]),
            n_jobs=1,
        )
        m.fit(Xtr[use_cols].values, ytr)
        preds = m.predict(Xva[use_cols].values)
        preds = np.clip(preds, 1e-6, 1.0 - 1e-6)
        oof_p_miss[va_idx] = preds

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    Xtr = X_tr_all.copy()
    Xte = test.copy()
    Xtr["kicker_te"] = fold_target_encode_miss(Xtr[cat], y_tr_all, Xtr[cat])
    Xte["kicker_te"] = fold_target_encode_miss(train[cat], y_tr_all, Xte[cat])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xtr[[cat]])
    Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
    Xte["kicker_id"] = enc.transform(Xte[[cat]]).astype("int64")
    use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

    final = SklearnModel(
        sublist=None,
        n_trees=int(BART_BEST["n_trees"]),
        n_chains=1,
        n_samples=int(BART_BEST["n_samples"]),
        n_burn=int(BART_BEST["n_burn"]),
        thin=float(BART_BEST["thin"]),
        n_jobs=1,
    )
    final.fit(Xtr[use_cols].values, y_tr_all)
    pte_miss_raw = final.predict(Xte[use_cols].values)
    pte_miss_raw = np.clip(pte_miss_raw, 1e-6, 1.0 - 1e-6)

    pte_miss = apply_isotonic_by_range(test["kick_distance"].values,
                                       pte_miss_raw, irs, bins=BINS)
    pte_miss = np.clip(pte_miss, 1e-6, 1.0 - 1e-6)
    y_te_make = 1 - y_te_miss
    pte_make = 1.0 - pte_miss
    return pte_make, y_te_make


def run_logreg(train, test):
    print("\n[Logistic Regression] Fitting...")
    X_tr_all, y_tr_all, num, cat = build_xy_make(train)
    X_te_raw, y_te, _, _ = build_xy_make(test)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_p = np.zeros(len(train))
    oof_y = y_tr_all.copy()
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y[tr_idx], oof_y[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["kicker_te"] = fold_target_encode_from_make(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_target_encode_from_make(Xtr[cat], ytr, Xva[cat])
        use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

        scaler = StandardScaler()
        Xtr_mat = scaler.fit_transform(Xtr[use_cols])
        Xva_mat = scaler.transform(Xva[use_cols])

        sw = compute_sw(ytr)
        clf = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=LR_BEST_C,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf.fit(Xtr_mat, ytr, sample_weight=sw)
        p_va = clf.predict_proba(Xva_mat)[:, 1]
        oof_p[va_idx] = p_va

    irs = fit_isotonic_by_range(oof_dist, oof_p, oof_y, bins=BINS)

    Xtr = X_tr_all.copy()
    Xte = test.copy()
    Xtr["kicker_te"] = fold_target_encode_from_make(Xtr[cat], y_tr_all, Xtr[cat])
    Xte["kicker_te"] = fold_target_encode_from_make(train[cat], y_tr_all, Xte[cat])
    use_cols = [c for c in Xtr.columns if c not in CATEGORICAL]

    scaler = StandardScaler()
    Xtr_mat_all = scaler.fit_transform(Xtr[use_cols])
    Xte_mat = scaler.transform(Xte[use_cols])

    final = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=LR_BEST_C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    final.fit(Xtr_mat_all, y_tr_all, sample_weight=compute_sw(y_tr_all))
    pte_raw = final.predict_proba(Xte_mat)[:, 1]
    pte = apply_isotonic_by_range(test["kick_distance"].values, pte_raw, irs, bins=BINS)
    pte = np.clip(pte, 1e-6, 1.0 - 1e-6)
    return pte, y_te


# ----------------- MAIN -----------------

def main():
    df = load_df()
    latest_season = int(df["season"].max())
    test = df[df["season"] == latest_season].reset_index(drop=True)
    train = df[df["season"] < latest_season].reset_index(drop=True)
    print(f"Train seasons ≤ {latest_season-1}: {len(train)} rows | "
          f"Test season {latest_season}: {len(test)} rows")

    # One script that runs all models and plots histograms
    models = []

    bag_p, bag_y = run_bagging(train, test)
    models.append(("Bagging", bag_p, bag_y))

    bayes_p, bayes_y = run_bayes_lr(train, test)
    models.append(("Bayes-LR", bayes_p, bayes_y))

    lgbm_p, lgbm_y = run_lgbm(train, test)
    models.append(("LightGBM", lgbm_p, lgbm_y))

    gam_p, gam_y = run_gam(train, test)
    models.append(("GAM", gam_p, gam_y))

    bart_p, bart_y = run_bart(train, test)
    models.append(("BART", bart_p, bart_y))

    lr_p, lr_y = run_logreg(train, test)
    models.append(("LogReg", lr_p, lr_y))

    # Plot histograms + print quick sanity metrics
    for name, p_make, y_true in models:
        brier = brier_score_loss(y_true, p_make)
        auc = roc_auc_score(y_true, p_make)
        y_miss = 1 - y_true
        p_miss = 1 - p_make
        pr_miss = average_precision_score(y_miss, p_miss)
        print(f"\n[{name}] TEST metrics (from this script)")
        print(f"  Brier (P(make)) = {brier:.5f}")
        print(f"  AUC (P(make))   = {auc:.4f}")
        print(f"  PR-AUC(miss)    = {pr_miss:.4f}")

        plot_hist(name, p_make, y_true, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
