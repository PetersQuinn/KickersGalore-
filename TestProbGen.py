# kicker_save_probs_all_models.py
#
# Re-train tuned models, calibrate, and save TEST probabilities for ensembling.
# Outputs:
#   model_probs/probs_bagging.csv
#   model_probs/probs_bayes_lr.csv
#   model_probs/probs_lgbm.csv
#   model_probs/probs_gam.csv
#   model_probs/probs_bart.csv
#   model_probs/probs_lr.csv
#
# Each CSV has:
#   y_te   : 1=make, 0=miss
#   p_make : calibrated P(make) on TEST

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from pygam import LogisticGAM
from gbart.modified_bartpy.sklearnmodel import SklearnModel

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TARGET = "field_goal_result_binary"  # 1=make, 0=miss
CATEGORICAL = ["kicker_player_name"]

PARQUET = "field_goals_model_ready.parquet"
CSV = "field_goals_model_ready.csv"

N_ECE_BINS = 10
BINS = (0, 40, 50, 80)  # distance bins for per-range isotonic

OUT_DIR = Path("model_probs")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


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
    Class-balancing sample weights for binary labels in {0,1}.
    """
    y = pd.Series(y)
    cnt = y.value_counts()
    tot = len(y)
    return y.map({c: tot / (2 * cnt[c]) for c in cnt.index}).values


def fold_te_from_miss(train_col, y_miss, valid_col, smoothing=20.0):
    """
    Target encoding for kicker miss rate when y_miss: 1=miss, 0=make.
    """
    miss = (y_miss == 1).astype(int)
    prior = miss.mean()
    g = (
        pd.DataFrame({"k": train_col.values, "miss": miss})
        .groupby("k")["miss"]
        .agg(["mean", "count"])
    )
    enc = (g["count"] * g["mean"] + smoothing * prior) / (g["count"] + smoothing)
    return valid_col.map(enc).fillna(prior).values


def fold_te_from_make(train_col, y_make, valid_col, smoothing=20.0):
    """
    Target encoding for kicker miss rate when y_make: 1=make, 0=miss.
    """
    miss = (y_make == 0).astype(int)
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
    p: predicted probabilities
    y: true labels for that probability (same semantics as p)
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


def build_xy_miss(df):
    """
    For models trained on y_miss: 1=miss, 0=make
    """
    y_make = df[TARGET].astype(int).values  # 1=make, 0=miss
    y_miss = (1 - y_make).astype(int)
    X = df.drop(columns=[TARGET]).copy()
    num = [c for c in X.columns if c not in CATEGORICAL]
    cat = CATEGORICAL[0]
    return X, y_miss, num, cat


def build_xy_make(df):
    """
    For models trained on y_make: 1=make, 0=miss
    """
    y_make = df[TARGET].astype(int).values
    X = df.drop(columns=[TARGET]).copy()
    num = [c for c in X.columns if c not in CATEGORICAL]
    cat = CATEGORICAL[0]
    return X, y_make, num, cat


# ---------------------------------------------------------------------
# Bagging (DecisionTree + BaggingClassifier; tuned params)
# ---------------------------------------------------------------------


def save_probs_bagging(train, test):
    print("\n[Bagging] Fitting + calibrating...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # tuned params you gave
    bag_params = dict(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        max_samples=0.9,
        max_features=0.8,
    )

    # OOF predictions for isotonic (P(miss))
    oof_p = np.zeros(len(train))
    oof_y = y_tr_all
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y[tr_idx], oof_y[va_idx]

        # kicker TE + id
        Xtr = Xtr.copy()
        Xva = Xva.copy()
        Xtr["kicker_te"] = fold_te_from_miss(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_te_from_miss(Xtr[cat], ytr, Xva[cat])

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(Xtr[[cat]])
        Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
        Xva["kicker_id"] = enc.transform(Xva[[cat]]).astype("int64")

        use_cols = num + ["kicker_te", "kicker_id"]
        Xtr_use = Xtr[use_cols].values
        Xva_use = Xva[use_cols].values

        base = DecisionTreeClassifier(
            max_depth=bag_params["max_depth"],
            min_samples_leaf=bag_params["min_samples_leaf"],
            random_state=RANDOM_STATE,
        )
        clf = BaggingClassifier(
            estimator=base,
            n_estimators=bag_params["n_estimators"],
            max_samples=bag_params["max_samples"],
            max_features=bag_params["max_features"],
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf.fit(Xtr_use, ytr)
        oof_p[va_idx] = clf.predict_proba(Xva_use)[:, 1]  # P(miss)

    irs = fit_isotonic_by_range(oof_dist, oof_p, oof_y, bins=BINS)

    # train on all train
    Xtr = X_tr_all.copy()
    Xtr["kicker_te"] = fold_te_from_miss(Xtr[cat], y_tr_all, Xtr[cat])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xtr[[cat]])
    Xtr["kicker_id"] = enc.transform(Xtr[[cat]]).astype("int64")
    use_cols = num + ["kicker_te", "kicker_id"]
    Xtr_use = Xtr[use_cols].values

    base = DecisionTreeClassifier(
        max_depth=bag_params["max_depth"],
        min_samples_leaf=bag_params["min_samples_leaf"],
        random_state=RANDOM_STATE,
    )
    final = BaggingClassifier(
        estimator=base,
        n_estimators=bag_params["n_estimators"],
        max_samples=bag_params["max_samples"],
        max_features=bag_params["max_features"],
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    final.fit(Xtr_use, y_tr_all)

    # test encoding
    te = test.copy()
    te["kicker_te"] = fold_te_from_miss(
        train[cat], y_tr_all, test[cat]
    )
    te["kicker_id"] = enc.transform(test[[cat]]).astype("int64")
    Xte_use = te[use_cols].values

    p_miss_raw = final.predict_proba(Xte_use)[:, 1]
    p_miss = apply_isotonic_by_range(
        test["kick_distance"].values, p_miss_raw, irs, bins=BINS
    )
    p_miss = np.clip(p_miss, 1e-6, 1 - 1e-6)

    y_te_make = 1 - y_te_miss
    p_make = 1.0 - p_miss

    out = pd.DataFrame({"y_te": y_te_make, "p_make": p_make})
    out.to_csv(OUT_DIR / "probs_bagging.csv", index=False)
    print("[Bagging] Saved to", OUT_DIR / "probs_bagging.csv")


# ---------------------------------------------------------------------
# Bayes-style Logistic Regression (just LR with tuned C)
# ---------------------------------------------------------------------


def _save_probs_logistic_like(train, test, C, out_name):
    """
    Shared logistic-style pipeline: TE + StandardScaler + LR on y_make.
    """
    X_tr_all, y_tr_all, num, cat = build_xy_make(train)
    X_te_raw, y_te, _, _ = build_xy_make(test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # OOF for isotonic (P(make))
    oof_p = np.zeros(len(train))
    oof_y = y_tr_all
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y[tr_idx], oof_y[va_idx]

        Xtr = Xtr.copy()
        Xva = Xva.copy()
        # kicker TE (miss rate given y_make)
        Xtr["kicker_te"] = fold_te_from_make(Xtr[cat], ytr, Xtr[cat])
        Xva["kicker_te"] = fold_te_from_make(Xtr[cat], ytr, Xva[cat])

        use_cols = num + ["kicker_te"]
        scaler = StandardScaler()
        Xtr_mat = scaler.fit_transform(Xtr[use_cols])
        Xva_mat = scaler.transform(Xva[use_cols])

        sw = compute_sw(ytr)

        clf = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=C,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf.fit(Xtr_mat, ytr, sample_weight=sw)
        oof_p[va_idx] = clf.predict_proba(Xva_mat)[:, 1]  # P(make)

    irs = fit_isotonic_by_range(oof_dist, oof_p, oof_y, bins=BINS)

    # Train on all train
    Xtr = X_tr_all.copy()
    Xtr["kicker_te"] = fold_te_from_make(Xtr[cat], y_tr_all, Xtr[cat])
    use_cols = num + ["kicker_te"]
    scaler = StandardScaler()
    Xtr_mat_all = scaler.fit_transform(Xtr[use_cols])

    test_enc = test.copy()
    test_enc["kicker_te"] = fold_te_from_make(
        train[cat], y_tr_all, test[cat]
    )
    Xte_mat = scaler.transform(test_enc[use_cols])

    final = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    final.fit(Xtr_mat_all, y_tr_all, sample_weight=compute_sw(y_tr_all))
    p_make_raw = final.predict_proba(Xte_mat)[:, 1]
    p_make = apply_isotonic_by_range(
        test["kick_distance"].values, p_make_raw, irs, bins=BINS
    )
    p_make = np.clip(p_make, 1e-6, 1 - 1e-6)

    out = pd.DataFrame({"y_te": y_te, "p_make": p_make})
    out.to_csv(OUT_DIR / out_name, index=False)
    print(f"[{out_name}] Saved to", OUT_DIR / out_name)


def save_probs_bayes_lr(train, test):
    # Tuned: C=0.001, penalty="l2"
    _save_probs_logistic_like(train, test, C=0.001, out_name="probs_bayes_lr.csv")


def save_probs_lr(train, test):
    # Tuned: C=0.00126743
    _save_probs_logistic_like(train, test, C=0.00126743, out_name="probs_lr.csv")


# ---------------------------------------------------------------------
# LightGBM (tuned params)
# ---------------------------------------------------------------------


def save_probs_lgbm(train, test):
    print("\n[LightGBM] Fitting + calibrating...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_params = {
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

    # OOF P(miss)
    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr_enc, Xva_enc = _lgbm_add_encodings(Xtr, Xva, ytr, num, cat)
        sw = compute_sw(ytr)

        pos = ytr.sum()
        neg = (ytr == 0).sum()
        spw = max(1.0, neg / max(1, pos))

        mono = [0] * len(Xtr_enc.columns)
        if "kick_distance" in Xtr_enc.columns:
            mono[list(Xtr_enc.columns).index("kick_distance")] = 1

        m = lgb.LGBMClassifier(
            objective="binary",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=spw,
            monotone_constraints=mono,
            verbosity=-1,
            **best_params,
        )
        m.fit(
            Xtr_enc,
            ytr,
            sample_weight=sw,
            eval_set=[(Xva_enc, yva)],
            eval_metric="auc",
            callbacks=[lgb.log_evaluation(0)],
        )
        oof_p_miss[va_idx] = m.predict_proba(Xva_enc)[:, 1]

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    # train on all train
    Xtr_enc, _ = _lgbm_add_encodings(X_tr_all, X_tr_all, y_tr_all, num, cat)

    test_enc = test.copy()
    test_enc["kicker_te"] = fold_te_from_miss(
        train[CATEGORICAL[0]], y_tr_all, test[CATEGORICAL[0]]
    )
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[[CATEGORICAL[0]]])
    test_enc["kicker_id"] = enc.transform(test[[CATEGORICAL[0]]]).astype("int64")
    use_cols = list(Xtr_enc.columns)
    Xte_enc = test_enc[use_cols]

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
        **best_params,
    )
    final.fit(
        Xtr_enc,
        y_tr_all,
        sample_weight=compute_sw(y_tr_all),
        callbacks=[lgb.log_evaluation(0)],
    )

    p_miss_raw = final.predict_proba(Xte_enc)[:, 1]
    p_miss = apply_isotonic_by_range(
        test["kick_distance"].values, p_miss_raw, irs, bins=BINS
    )
    p_miss = np.clip(p_miss, 1e-6, 1 - 1e-6)

    y_te_make = 1 - y_te_miss
    p_make = 1.0 - p_miss

    out = pd.DataFrame({"y_te": y_te_make, "p_make": p_make})
    out.to_csv(OUT_DIR / "probs_lgbm.csv", index=False)
    print("[LightGBM] Saved to", OUT_DIR / "probs_lgbm.csv")


def _lgbm_add_encodings(X_tr, X_va, y_tr, num, cat):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_tr["kicker_te"] = fold_te_from_miss(X_tr[cat], y_tr, X_tr[cat])
    X_va["kicker_te"] = fold_te_from_miss(X_tr[cat], y_tr, X_va[cat])

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X_tr[[cat]])
    X_tr["kicker_id"] = enc.transform(X_tr[[cat]]).astype("int64")
    X_va["kicker_id"] = enc.transform(X_va[[cat]]).astype("int64")
    use = num + ["kicker_te", "kicker_id"]
    return X_tr[use], X_va[use]


# ---------------------------------------------------------------------
# GAM (pygam LogisticGAM, tuned lam / n_splines)
# ---------------------------------------------------------------------


def save_probs_gam(train, test):
    print("\n[GAM] Fitting + calibrating...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_lam = 1000.0
    n_splines = 8

    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr_mat, Xva_mat, use_cols, scaler = _gam_add_te_and_scale(
            Xtr, Xva, ytr, num, cat
        )
        sw = compute_sw(ytr)

        gam = LogisticGAM(
            lam=best_lam,
            n_splines=n_splines,
            max_iter=300,
            fit_intercept=True,
        )
        try:
            gam.fit(Xtr_mat, ytr, weights=sw)
            p_fold = gam.predict_proba(Xva_mat)
        except Exception as e:
            print(f"[GAM] OOF fold crashed: {e}")
            p_fold = np.full(len(yva), 0.5, dtype=float)

        p_fold = np.clip(p_fold, 1e-6, 1 - 1e-6)
        oof_p_miss[va_idx] = p_fold

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    # train on all train
    Xtr_mat_all, _, use_cols, scaler = _gam_add_te_and_scale(
        X_tr_all, X_tr_all, y_tr_all, num, cat
    )

    test_enc = test.copy()
    test_enc["kicker_te"] = fold_te_from_miss(
        train[CATEGORICAL[0]], y_tr_all, test[CATEGORICAL[0]]
    )
    Xte_mat = scaler.transform(test_enc[use_cols])

    gam_final = LogisticGAM(
        lam=best_lam,
        n_splines=n_splines,
        max_iter=300,
        fit_intercept=True,
    )
    try:
        gam_final.fit(Xtr_mat_all, y_tr_all, weights=compute_sw(y_tr_all))
        p_miss_raw = gam_final.predict_proba(Xte_mat)
    except Exception as e:
        print(f"[GAM] final fit crashed: {e}")
        p_miss_raw = np.full(len(y_te_miss), 0.5, dtype=float)

    p_miss_raw = np.clip(p_miss_raw, 1e-6, 1 - 1e-6)
    p_miss = apply_isotonic_by_range(
        test["kick_distance"].values, p_miss_raw, irs, bins=BINS
    )
    p_miss = np.clip(p_miss, 1e-6, 1 - 1e-6)

    y_te_make = 1 - y_te_miss
    p_make = 1.0 - p_miss

    out = pd.DataFrame({"y_te": y_te_make, "p_make": p_make})
    out.to_csv(OUT_DIR / "probs_gam.csv", index=False)
    print("[GAM] Saved to", OUT_DIR / "probs_gam.csv")


def _gam_add_te_and_scale(X_tr, X_va, y_tr, num_cols, cat):
    X_tr = X_tr.copy()
    X_va = X_va.copy()

    X_tr["kicker_te"] = fold_te_from_miss(X_tr[cat], y_tr, X_tr[cat])
    X_va["kicker_te"] = fold_te_from_miss(X_tr[cat], y_tr, X_va[cat])

    use_cols = num_cols + ["kicker_te"]
    scaler = StandardScaler()
    Xtr_mat = scaler.fit_transform(X_tr[use_cols])
    Xva_mat = scaler.transform(X_va[use_cols])
    return Xtr_mat, Xva_mat, use_cols, scaler


# ---------------------------------------------------------------------
# BART (gbart SklearnModel, tuned params)
# ---------------------------------------------------------------------


def save_probs_bart(train, test):
    print("\n[BART] Fitting + calibrating (this may be slow)...")
    X_tr_all, y_tr_all, num, cat = build_xy_miss(train)
    X_te_raw, y_te_miss, _, _ = build_xy_miss(test)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_params = {
        "n_trees": 200,
        "n_samples": 150,
        "n_burn": 400,
        "thin": 0.5,
    }

    oof_p_miss = np.zeros(len(train))
    oof_y_miss = y_tr_all
    oof_dist = train["kick_distance"].values

    for tr_idx, va_idx in skf.split(X_tr_all, oof_y_miss):
        Xtr, Xva = X_tr_all.iloc[tr_idx], X_tr_all.iloc[va_idx]
        ytr, yva = oof_y_miss[tr_idx], oof_y_miss[va_idx]

        Xtr_enc, Xva_enc = _bart_add_encodings(Xtr, Xva, ytr, num, cat)

        m = SklearnModel(
            sublist=None,
            n_trees=int(best_params["n_trees"]),
            n_chains=1,
            n_samples=int(best_params["n_samples"]),
            n_burn=int(best_params["n_burn"]),
            thin=float(best_params["thin"]),
            n_jobs=1,
        )
        m.fit(Xtr_enc.values, ytr)
        preds = m.predict(Xva_enc.values)
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
        oof_p_miss[va_idx] = preds

    irs = fit_isotonic_by_range(oof_dist, oof_p_miss, oof_y_miss, bins=BINS)

    # train on all
    Xtr_enc, _ = _bart_add_encodings(X_tr_all, X_tr_all, y_tr_all, num, cat)

    test_enc = test.copy()
    test_enc["kicker_te"] = fold_te_from_miss(
        train[CATEGORICAL[0]], y_tr_all, test[CATEGORICAL[0]]
    )
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[[CATEGORICAL[0]]])
    test_enc["kicker_id"] = enc.transform(test[[CATEGORICAL[0]]]).astype("int64")
    use_cols = list(Xtr_enc.columns)
    Xte_enc = test_enc[use_cols]

    final = SklearnModel(
        sublist=None,
        n_trees=int(best_params["n_trees"]),
        n_chains=1,
        n_samples=int(best_params["n_samples"]),
        n_burn=int(best_params["n_burn"]),
        thin=float(best_params["thin"]),
        n_jobs=1,
    )
    final.fit(Xtr_enc.values, y_tr_all)
    p_miss_raw = final.predict(Xte_enc.values)
    p_miss_raw = np.clip(p_miss_raw, 1e-6, 1 - 1e-6)

    p_miss = apply_isotonic_by_range(
        test["kick_distance"].values, p_miss_raw, irs, bins=BINS
    )
    p_miss = np.clip(p_miss, 1e-6, 1 - 1e-6)

    y_te_make = 1 - y_te_miss
    p_make = 1.0 - p_miss

    out = pd.DataFrame({"y_te": y_te_make, "p_make": p_make})
    out.to_csv(OUT_DIR / "probs_bart.csv", index=False)
    print("[BART] Saved to", OUT_DIR / "probs_bart.csv")


def _bart_add_encodings(X_tr, X_va, y_tr, num, cat):
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_tr["kicker_te"] = fold_te_from_miss(X_tr[cat], y_tr, X_tr[cat])
    X_va["kicker_te"] = fold_te_from_miss(X_tr[cat], y_tr, X_va[cat])
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(X_tr[[cat]])
    X_tr["kicker_id"] = enc.transform(X_tr[[cat]]).astype("int64")
    X_va["kicker_id"] = enc.transform(X_va[[cat]]).astype("int64")
    use = num + ["kicker_te", "kicker_id"]
    return X_tr[use], X_va[use]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    df = load_df()
    latest_season = int(df["season"].max())
    test = df[df["season"] == latest_season].reset_index(drop=True)
    train = df[df["season"] < latest_season].reset_index(drop=True)
    print(
        f"Train seasons ≤ {latest_season-1}: {len(train)} rows | "
        f"Test season {latest_season}: {len(test)} rows"
    )

    save_probs_bagging(train, test)
    save_probs_bayes_lr(train, test)
    save_probs_lgbm(train, test)
    save_probs_gam(train, test)
    save_probs_bart(train, test)
    save_probs_lr(train, test)

    print("\nAll model probability CSVs saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
