import os
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

from pygam import LogisticGAM
from gbart.modified_bartpy.sklearnmodel import SklearnModel  # BART
import lightgbm as lgb

# -------------------- Config --------------------

RANDOM_STATE = 42
TARGET = "field_goal_result_binary"  # 1=make, 0=miss
CATEGORICAL = ["kicker_player_name"]
PARQUET = "field_goals_model_ready.parquet"
CSV = "field_goals_model_ready.csv"

# distance bins for optional calibration (kept simple; one global isotonic)
BINS = (0, 40, 50, 80)

# causal features from BART table
CAUSAL_FEATURES = [
    "kick_distance",
    "is_snow",
    "is_rain",
    "roof_binary",
    "buzzer_beater_binary",  # we'll handle fallback name below
    "career_fg_pct",
]

# folder to save stacking data
STACK_DIR = "stack_meta"
os.makedirs(STACK_DIR, exist_ok=True)


# -------------------- Helpers --------------------

def load_df():
    if Path(PARQUET).exists():
        try:
            print(f"[load] Loading PARQUET: {PARQUET}")
            return pd.read_parquet(PARQUET)
        except Exception as e:
            print(f"[WARN] Failed to read {PARQUET} ({e}). Falling back to CSV.")
    print(f"[load] Loading CSV: {CSV}")
    return pd.read_csv(CSV)


def ece_score(probs, y, n_bins=10):
    """Expected Calibration Error for P(make) vs y (1=make)."""
    probs = np.asarray(probs)
    y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        e += m.sum() / N * abs(probs[m].mean() - y[m].mean())
    return float(e)


def fit_isotonic_global(p_train, y_train):
    """
    Fit a single isotonic regressor mapping raw P(make) -> calibrated P(make)
    using all train data. (Simpler than per-distance bins.)
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_train, y_train)
    return ir


def apply_isotonic_global(p, ir):
    p = np.asarray(p, dtype=float)
    return ir.transform(p)


def prepare_features(df):
    """Split into X (drop target & kicker name), y_make."""
    y_make = df[TARGET].astype(int).values  # 1=make, 0=miss
    X = df.drop(columns=[TARGET] + CATEGORICAL, errors="ignore").copy()
    return X, y_make


def ensure_causal_columns(df):
    """Make sure the 6 causal features exist, with a buzzer_beater fallback."""
    cols = list(df.columns)
    if "buzzer_beater_binary" not in cols and "buzzer_beatery_binary" in cols:
        df["buzzer_beater_binary"] = df["buzzer_beatery_binary"]
    missing = [c for c in CAUSAL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing causal feature columns in df: {missing}")


# -------------------- Base model trainers --------------------

def train_predict_bagging(X_tr, y_tr, X_te):
    print("[bagging] Training...")
    base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
    )
    model = BaggingClassifier(
        estimator=base,
        n_estimators=600,
        max_samples=0.9,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print("[bagging] Predicting train/test...")
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    return p_tr, p_te


def train_predict_bayes_lr(X_tr, y_tr, X_te):
    """
    'Bayesian-style' LR here is approximated as strongly-regularized LR
    with C=0.001 and L2 penalty (matches your tuned params).
    """
    print("[bayes_lr] Training...")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)

    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=0.001,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    clf.fit(Xtr_s, y_tr)
    print("[bayes_lr] Predicting train/test...")
    p_tr = clf.predict_proba(Xtr_s)[:, 1]
    p_te = clf.predict_proba(Xte_s)[:, 1]
    return p_tr, p_te


def train_predict_lgbm(X_tr, y_tr, X_te):
    print("[lgbm] Training...")
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=127,
        learning_rate=0.015,
        min_child_samples=10,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        n_estimators=1800,
        max_depth=-1,
        min_split_gain=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print("[lgbm] Predicting train/test...")
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]
    return p_tr, p_te


def train_predict_gam(X_tr, y_tr, X_te):
    print("[gam] Training...")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)

    gam = LogisticGAM(
        lam=1000.0,
        n_splines=8,
        max_iter=300,
        fit_intercept=True,
    )
    gam.fit(Xtr_s, y_tr)
    print("[gam] Predicting train/test...")
    p_tr = gam.predict_proba(Xtr_s)
    p_te = gam.predict_proba(Xte_s)
    return p_tr, p_te


def train_predict_bart(X_tr, y_tr, X_te):
    print("[bart] Training (this may take a while)...")
    model = SklearnModel(
        sublist=None,
        n_trees=200,
        n_chains=1,
        n_samples=150,
        n_burn=400,
        thin=0.5,
        n_jobs=1,  # safer on Windows
    )
    model.fit(X_tr.values, y_tr)
    print("[bart] Predicting train/test...")
    p_tr = model.predict(X_tr.values)
    p_te = model.predict(X_te.values)
    # clip to (0,1)
    p_tr = np.clip(p_tr, 1e-6, 1 - 1e-6)
    p_te = np.clip(p_te, 1e-6, 1 - 1e-6)
    return p_tr, p_te


def train_predict_lr(X_tr, y_tr, X_te):
    print("[lr] Training...")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)

    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=0.00126743,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )
    clf.fit(Xtr_s, y_tr)
    print("[lr] Predicting train/test...")
    p_tr = clf.predict_proba(Xtr_s)[:, 1]
    p_te = clf.predict_proba(Xte_s)[:, 1]
    return p_tr, p_te


# -------------------- Main pipeline --------------------

def main():
    df = load_df()
    ensure_causal_columns(df)

    latest_season = int(df["season"].max())
    test_df = df[df["season"] == latest_season].reset_index(drop=True)
    train_df = df[df["season"] < latest_season].reset_index(drop=True)
    print(
        f"[split] Train seasons ≤ {latest_season-1}: {len(train_df)} rows | "
        f"Test season {latest_season}: {len(test_df)} rows"
    )

    # features / targets for base models (drop target + kicker name)
    X_tr, y_tr = prepare_features(train_df)
    X_te, y_te = prepare_features(test_df)

    # dictionary to collect base model probs
    base_train = {}
    base_test = {}

    # 1. Bagging
    p_tr, p_te = train_predict_bagging(X_tr, y_tr, X_te)
    base_train["bagging"] = p_tr
    base_test["bagging"] = p_te

    # 2. Bayes-LR-ish
    p_tr, p_te = train_predict_bayes_lr(X_tr, y_tr, X_te)
    base_train["bayes_lr"] = p_tr
    base_test["bayes_lr"] = p_te

    # 3. LightGBM
    p_tr, p_te = train_predict_lgbm(X_tr, y_tr, X_te)
    base_train["lgbm"] = p_tr
    base_test["lgbm"] = p_te

    # 4. GAM
    p_tr, p_te = train_predict_gam(X_tr, y_tr, X_te)
    base_train["gam"] = p_tr
    base_test["gam"] = p_te

    # 5. BART
    p_tr, p_te = train_predict_bart(X_tr, y_tr, X_te)
    base_train["bart"] = p_tr
    base_test["bart"] = p_te

    # 6. Logistic Regression
    p_tr, p_te = train_predict_lr(X_tr, y_tr, X_te)
    base_train["lr"] = p_tr
    base_test["lr"] = p_te

    # Optional: simple global isotonic calibration per model (fit on train, apply to both)
    print("[calibration] Fitting global isotonic calibrators per model...")
    for name in base_train.keys():
        print(f"  - {name}")
        ir = fit_isotonic_global(base_train[name], y_tr)
        base_train[name] = apply_isotonic_global(base_train[name], ir)
        base_test[name] = apply_isotonic_global(base_test[name], ir)

    # ---------------- Build stacking datasets ----------------

    print("[stack] Building stack_train and stack_test DataFrames...")

    model_names = ["bagging", "bayes_lr", "lgbm", "gam", "bart", "lr"]

    stack_train = pd.DataFrame({"y": y_tr})
    stack_test = pd.DataFrame({"y": y_te})

    for m in model_names:
        stack_train[f"p_{m}"] = base_train[m]
        stack_test[f"p_{m}"] = base_test[m]

    # add causal features
    for feat in CAUSAL_FEATURES:
        stack_train[feat] = train_df[feat].values
        stack_test[feat] = test_df[feat].values

    # Save for inspection
    train_path = os.path.join(STACK_DIR, "stack_train.csv")
    test_path = os.path.join(STACK_DIR, "stack_test.csv")
    stack_train.to_csv(train_path, index=False)
    stack_test.to_csv(test_path, index=False)
    print(f"[stack] Saved stack_train to {train_path}")
    print(f"[stack] Saved stack_test to {test_path}")

    # ---------------- Meta-LR ensembles with feature subsets ----------------

    print("\n---------------- Meta-Logistic Regression Ensembles ----------------")

    # base-probability columns (always included in meta model)
    prob_cols = [f"p_{m}" for m in model_names]

    results = []

    total_subsets = 2 ** len(CAUSAL_FEATURES)
    subset_idx = 0

    for r in range(len(CAUSAL_FEATURES) + 1):
        for feat_subset in itertools.combinations(CAUSAL_FEATURES, r):
            subset_idx += 1
            feat_subset = list(feat_subset)
            print(
                f"[meta-LR] ({subset_idx}/{total_subsets}) "
                f"Training with causal features: {feat_subset if feat_subset else 'NONE'}"
            )

            cols = prob_cols + feat_subset

            Xtr_meta = stack_train[cols].values
            Xte_meta = stack_test[cols].values
            ytr_meta = stack_train["y"].values
            yte_meta = stack_test["y"].values

            # scale meta features
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr_meta)
            Xte_s = scaler.transform(Xte_meta)

            meta_clf = LogisticRegression(
                solver="liblinear",
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1000,
            )
            meta_clf.fit(Xtr_s, ytr_meta)
            pte_make = meta_clf.predict_proba(Xte_s)[:, 1]

            # metrics
            brier = brier_score_loss(yte_meta, pte_make)
            auc = roc_auc_score(yte_meta, pte_make)
            y_miss = 1 - yte_meta
            p_miss = 1 - pte_make
            pr_miss = average_precision_score(y_miss, p_miss)
            ece10 = ece_score(pte_make, yte_meta, n_bins=10)
            thr = 0.5
            yhat = (pte_make >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte_meta, yhat).ravel()

            name = (
                f"MetaLR[probs"
                + ("" if not feat_subset else "+" + "+".join(feat_subset))
                + "]"
            )

            results.append(
                {
                    "ensemble_name": name,
                    "causal_features": feat_subset,
                    "Brier": brier,
                    "AUC": auc,
                    "PR_miss": pr_miss,
                    "ECE10": ece10,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )

    # sort by Brier
    results_df = pd.DataFrame(results).sort_values("Brier").reset_index(drop=True)

    print("\nTop 20 meta-LR ensembles by Brier:")
    print(
        "rank | ensemble_name                                 | Brier    | AUC    | PR-miss | ECE@10"
    )
    print("-" * 90)
    for i in range(min(20, len(results_df))):
        row = results_df.iloc[i]
        print(
            f"{i+1:4d} | {row['ensemble_name'][:43]:43s} | "
            f"{row['Brier']:.5f} | {row['AUC']:.4f} | "
            f"{row['PR_miss']:.4f} | {row['ECE10']:.4f}"
        )

    best = results_df.iloc[0]
    print("\n==============================================================")
    print("Best meta-LR ensemble by Brier:")
    print(best["ensemble_name"])
    print("Causal features:", best["causal_features"])
    print(
        f"\n=== TEST — {best['ensemble_name']} ===\n"
        f"Brier={best['Brier']:.5f} (primary, on P(make))\n"
        f"AUC={best['AUC']:.4f} | PR-AUC(miss)={best['PR_miss']:.4f} | "
        f"ECE@10={best['ECE10']:.4f}\n"
        f"Threshold for confusion matrix (on P(make)): 0.50\n"
        f"Confusion matrix (1=make, 0=miss): "
        f"tn={best['tn']} fp={best['fp']} fn={best['fn']} tp={best['tp']}"
    )

    # Optional: save full results table
    results_path = os.path.join(STACK_DIR, "meta_lr_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[meta-LR] Saved full results to {results_path}")


if __name__ == "__main__":
    main()
