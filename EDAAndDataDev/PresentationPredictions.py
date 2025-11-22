"""
Train the weighted ensemble model and predict P(make) for 3 demo kicks.

Model components:
- Bagging (DecisionTree base)
- LightGBM
- GAM (LogisticGAM)
- Logistic Regression

Ensemble weights:
0.10 * Bagging + 0.35 * LGBM + 0.40 * GAM + 0.15 * LR
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from lightgbm import LGBMClassifier
from pygam import LogisticGAM


CSV_PATH = "field_goals_model_ready.csv"

FEATURE_COLS = [
    "score_differential",
    "kick_distance",
    "temp",
    "wind",
    "season_type_binary",
    "roof_binary",
    "surface_binary",
    "altitude",
    "vegas_wp_effective",
    "is_rain",
    "is_snow",
    "career_attempts",
    "career_fg_pct",
    "is_4th_qtr",
    "buzzer_beater_binary",
]

TARGET_COL = "field_goal_result_binary"


DIST_BUCKETS = [
    (0, 29, "0-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
    (50, 120, "50+"),  
]


def get_distance_bucket_label(d):
    for lo, hi, label in DIST_BUCKETS:
        if lo <= d <= hi:
            return label
    return "50+"  # fallback


def fit_per_distance_isotonic(distances, raw_probs, y, min_samples=30):
    distances = np.asarray(distances)
    raw_probs = np.asarray(raw_probs)
    y = np.asarray(y)

    # Global calibrator as fallback
    global_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    global_iso.fit(raw_probs, y)

    per_bucket = {}
    for lo, hi, label in DIST_BUCKETS:
        mask = (distances >= lo) & (distances <= hi)
        if mask.sum() >= min_samples:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_probs[mask], y[mask])
            per_bucket[label] = iso

    return {"global": global_iso, "per_bucket": per_bucket}


def apply_per_distance_isotonic(calib, distances, raw_probs):
    distances = np.asarray(distances)
    raw_probs = np.asarray(raw_probs)
    calibrated = np.zeros_like(raw_probs, dtype=float)

    for i in range(len(raw_probs)):
        d = distances[i]
        label = get_distance_bucket_label(d)
        iso = calib["per_bucket"].get(label, calib["global"])
        calibrated[i] = iso.predict([raw_probs[i]])[0]

    return calibrated

def train_models(df):
    if (df["season"] == 2024).any():
        train_mask = df["season"] < 2024
    else:
        train_mask = np.ones(len(df), dtype=bool)

    train_df = df.loc[train_mask].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    dist_train = train_df["kick_distance"].values

    bag_base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )
    bagging = BaggingClassifier(
        estimator=bag_base,
        n_estimators=600,
        max_samples=0.9,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42,
        n_jobs=-1,
    )
    bagging.fit(X_train, y_train)
    bagging_probs_train = bagging.predict_proba(X_train)[:, 1]
    bagging_calib = fit_per_distance_isotonic(dist_train, bagging_probs_train, y_train)

    lgbm = LGBMClassifier(
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
        objective="binary",
        random_state=42,
        n_jobs=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_probs_train = lgbm.predict_proba(X_train)[:, 1]
    lgbm_calib = fit_per_distance_isotonic(dist_train, lgbm_probs_train, y_train)

    gam = LogisticGAM(lam=1000.0, n_splines=8)
    gam.fit(X_train, y_train)
    gam_probs_train = gam.predict_proba(X_train)
    gam_calib = fit_per_distance_isotonic(dist_train, gam_probs_train, y_train)

    lr = LogisticRegression(
        C=0.00126743,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    lr_probs_train = lr.predict_proba(X_train)[:, 1]
    lr_calib = fit_per_distance_isotonic(dist_train, lr_probs_train, y_train)

    models = {
        "bagging": {"model": bagging, "calib": bagging_calib},
        "lgbm": {"model": lgbm, "calib": lgbm_calib},
        "gam": {"model": gam, "calib": gam_calib},
        "lr": {"model": lr, "calib": lr_calib},
    }
    return models


def predict_for_kicks(models, kicks_df):
    X_new = kicks_df[FEATURE_COLS].values
    distances = kicks_df["kick_distance"].values

    # Base model raw probabilities
    bagging_raw = models["bagging"]["model"].predict_proba(X_new)[:, 1]
    lgbm_raw = models["lgbm"]["model"].predict_proba(X_new)[:, 1]
    gam_raw = models["gam"]["model"].predict_proba(X_new)
    lr_raw = models["lr"]["model"].predict_proba(X_new)[:, 1]

    # Calibrated per-distance probabilities
    bagging_cal = apply_per_distance_isotonic(models["bagging"]["calib"], distances, bagging_raw)
    lgbm_cal = apply_per_distance_isotonic(models["lgbm"]["calib"], distances, lgbm_raw)
    gam_cal = apply_per_distance_isotonic(models["gam"]["calib"], distances, gam_raw)
    lr_cal = apply_per_distance_isotonic(models["lr"]["calib"], distances, lr_raw)

    # Weighted ensemble 
    ensemble = (
        0.10 * bagging_cal
        + 0.35 * lgbm_cal
        + 0.40 * gam_cal
        + 0.15 * lr_cal
    )

    results = kicks_df.copy()
    results["P_make_bagging"] = bagging_cal
    results["P_make_lgbm"] = lgbm_cal
    results["P_make_gam"] = gam_cal
    results["P_make_lr"] = lr_cal
    results["P_make_ensemble"] = ensemble

    return results


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    models = train_models(df)

    demo_kicks = [
        {
            "kick_id": "Kick 1 – 68 yards, indoor, down 6",
            "score_differential": -6,
            "kick_distance": 68,
            "temp": 68,
            "wind": 0,
            "season_type_binary": 0,
            "roof_binary": 1,
            "surface_binary": 0,
            "altitude": 2000,
            "vegas_wp_effective": 0.372,
            "is_rain": 0,
            "is_snow": 0,
            "career_attempts": 43,
            "career_fg_pct": 0.86,
            "is_4th_qtr": 0,
            "buzzer_beater_binary": 0,
        },
        {
            "kick_id": "Kick 2 – 43 yards, playoff, windy, GW try",
            "score_differential": -1,
            "kick_distance": 43,
            "temp": 39,
            "wind": 14,
            "season_type_binary": 1,
            "roof_binary": 0,
            "surface_binary": 0,
            "altitude": 580,
            "vegas_wp_effective": 0.654,
            "is_rain": 0,
            "is_snow": 0,
            "career_attempts": 121,
            "career_fg_pct": 0.843,
            "is_4th_qtr": 1,
            "buzzer_beater_binary": 1,
        },
        {
            "kick_id": "Kick 3 – 52 yards, cold playoff, GW try",
            "score_differential": 0,
            "kick_distance": 52,
            "temp": 35,
            "wind": 4,
            "season_type_binary": 1,
            "roof_binary": 0,
            "surface_binary": 0,
            "altitude": 385,
            "vegas_wp_effective": 0.806,
            "is_rain": 0,
            "is_snow": 0,
            "career_attempts": 40,
            "career_fg_pct": 0.875,
            "is_4th_qtr": 1,
            "buzzer_beater_binary": 1,
        },
    ]

    demo_df = pd.DataFrame(demo_kicks)

    # Ensure columns all present and ordered
    for col in FEATURE_COLS:
        if col not in demo_df.columns:
            raise ValueError(f"Missing feature column in demo kicks: {col}")

    results = predict_for_kicks(models, demo_df)

    pd.set_option("display.precision", 4)
    print("\n=== Demo Kick Probabilities (P(make)) ===\n")
    cols_to_show = [
        "kick_id",
        "kick_distance",
        "score_differential",
        "temp",
        "wind",
        "season_type_binary",
        "roof_binary",
        "vegas_wp_effective",
        "career_attempts",
        "career_fg_pct",
        "is_4th_qtr",
        "buzzer_beater_binary",
        "P_make_bagging",
        "P_make_lgbm",
        "P_make_gam",
        "P_make_lr",
        "P_make_ensemble",
    ]
    print(results[cols_to_show])
    print("\n(Probabilities are per-distance calibrated and then combined in the weighted ensemble.)")
