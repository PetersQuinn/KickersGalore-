# causal_forest_all_features_FIXED.py
"""
Causal Forest (econml) causal analysis for the kicker dataset — with refined RELATED_DROPS
and resilient CSV/Parquet loading.

- Outcome: miss = 1 - field_goal_result_binary
- One CF run per treatment with treatment-specific covariates to avoid post-treatment control
- Binary effects: E[Y(1) - Y(0)] via effect(X, T0=0, T1=1)
- Continuous effects: finite-difference ATE over DELTA via effect(X, T0=t, T1=t+delta), also report per-unit
- Writes CSV: causal_forest_ate_summary.csv

Install:
  pip install econml scikit-learn

Note: CF uses built-in cross-fitting; no scaling required for trees. Keep binaries as 0/1.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from econml.dml import CausalForestDML

RANDOM_STATE = 42
PARQUET = "field_goals_model_ready.parquet"
CSV     = "field_goals_model_ready.csv"
TARGET  = "field_goal_result_binary"  # 1=make, 0=miss → miss=1
ID_COLS = ["kicker_player_name", "season"]

# -------------------- treatment configuration --------------------
BINARY_TREATMENTS = [
    "roof_binary",
    "is_rain",
    "is_snow",
    "surface_binary",
    "season_type_binary",
    "is_4th_qtr",
    "buzzer_beater_binary",
]

CONT_TREATMENTS = [
    "kick_distance",
    "temp",
    "wind",
    "vegas_wp_effective",
    "career_attempts",
    "career_fg_pct",
    "altitude",
    "score_differential",
]

DELTA = {
    "kick_distance": 1.0,
    "temp": 5.0,
    "wind": 5.0,
    "vegas_wp_effective": 0.10,
    "career_attempts": 50.0,
    "career_fg_pct": 1.0,
    "altitude": 1000.0,
    "score_differential": 3.0,
}

# Known post-treatment/overlap relationships to avoid conditioning on
RELATED_DROPS = {
    # Roof is often determined by precipitation (and sometimes wind); drop precip to avoid blocking its effect
    "roof_binary": ["is_rain", "is_snow"],
    # When rain is the treatment, drop roof and snow (sibling precip) to avoid post-treatment conditioning
    "is_rain": ["roof_binary", "is_snow"],
    # When snow is the treatment, drop roof and rain
    "is_snow": ["roof_binary", "is_rain"],
    # Quarter vs. buzzer-beater are tightly linked; drop the other when one is the treatment
    "is_4th_qtr": ["buzzer_beater_binary"],
    "buzzer_beater_binary": ["is_4th_qtr"],
}

# -------------------- robust loader --------------------

def load_df():
    if Path(PARQUET).exists():
        try:
            return pd.read_parquet(PARQUET, engine="pyarrow")
        except Exception as e:
            print(f"[warn] Failed to read parquet with pyarrow: {e}")
            try:
                import fastparquet  # noqa: F401
                print("[info] Retrying with fastparquet engine...")
                return pd.read_parquet(PARQUET, engine="fastparquet")
            except Exception as e2:
                print(f"[warn] fastparquet also failed: {e2}")
    print("[info] Falling back to CSV load...")
    return pd.read_csv(CSV)

# -------------------- helpers --------------------

def build_y(df: pd.DataFrame) -> np.ndarray:
    return (1 - df[TARGET].astype(int).values).astype(int)


def covariates_for_treatment(treatment: str, all_cols: list) -> list:
    drops = set([TARGET] + ID_COLS + [treatment])
    drops.update(RELATED_DROPS.get(treatment, []))
    return [c for c in all_cols if c not in drops]


def fit_cf(X, T, y, binary: bool) -> CausalForestDML:
    model_y = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=RANDOM_STATE)
    if binary:
        model_t = LogisticRegression(max_iter=1000, solver="lbfgs")
    else:
        model_t = Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE)

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=None,
        discrete_treatment=binary,
        random_state=RANDOM_STATE,
    )
    cf.fit(y, T, X=X)
    return cf


def binary_ate(df: pd.DataFrame, y: np.ndarray, treatment: str, covs: list):
    X = df[covs].values
    T = df[treatment].astype(int).values
    cf = fit_cf(X, T, y, binary=True)
    eff = cf.effect(X, T0=0, T1=1)
    return float(np.mean(eff)), eff


def continuous_ate(df: pd.DataFrame, y: np.ndarray, treatment: str, covs: list, delta: float):
    X = df[covs].values
    T = df[treatment].astype(float).values
    cf = fit_cf(X, T, y, binary=False)
    eff_delta = cf.effect(X, T0=T, T1=T + delta)
    ate_delta = float(np.mean(eff_delta))
    per_unit = float(ate_delta / delta) if delta != 0 else float("nan")
    return ate_delta, per_unit, eff_delta


def main():
    df = load_df()
    y  = build_y(df)
    all_cols = list(df.columns)

    rows = []

    # Binary
    for t in BINARY_TREATMENTS:
        if t not in df.columns:
            continue
        covs = covariates_for_treatment(t, all_cols)
        ate, ite = binary_ate(df, y, t, covs)
        rows.append({
            "treatment": t,
            "type": "binary",
            "delta": 1.0,
            "ATE_delta": ate,
            "ATE_per_unit": ate,
            "n": len(df)
        })

    # Continuous
    for t in CONT_TREATMENTS:
        if t not in df.columns:
            continue
        covs = covariates_for_treatment(t, all_cols)
        d = DELTA.get(t, 1.0)
        ate_d, per_unit, ite = continuous_ate(df, y, t, covs, d)
        rows.append({
            "treatment": t,
            "type": "continuous",
            "delta": d,
            "ATE_delta": ate_d,
            "ATE_per_unit": per_unit,
            "n": len(df)
        })

    out = pd.DataFrame(rows).sort_values(["type", "ATE_per_unit"], ascending=[True, True])
    print("\n=== Causal Forest Average Treatment Effects (miss probability) ===")
    print(out.to_string(index=False, justify="left", max_colwidth=20))

    out_path = Path("causal_forest_ate_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved summary to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
