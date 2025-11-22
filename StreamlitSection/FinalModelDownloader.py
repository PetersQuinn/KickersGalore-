"""
- Input:
    field_goals_model_ready.csv with columns:
    season, score_differential, kicker_player_name, kick_distance, temp, wind,
    season_type_binary, field_goal_result_binary, roof_binary, surface_binary,
    altitude, vegas_wp_effective, is_rain, is_snow, career_attempts,
    career_fg_pct, is_4th_qtr, buzzer_beater_binary

- Output:
    weighted_ensemble.pkl  (joblib dump of an sklearn-style estimator with
    predict_proba(X) -> [n_samples, 2] for classes [0=miss, 1=make])
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from pygam import LogisticGAM
from scipy import sparse 

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "field_goals_model_ready.csv"

OUTPUT_PKL = "weighted_ensemble.pkl"

TARGET_COL = "field_goal_result_binary"

FEATURE_COLS = [
    "season",
    "score_differential",
    "kicker_player_name",
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

CATEGORICAL_COLS = ["kicker_player_name"]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

ENSEMBLE_WEIGHTS = {
    "bagging": 0.10,
    "lgbm": 0.35,
    "gam": 0.40,
    "lr": 0.15,
}


class GamWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, lam=1000.0, n_splines=8):
        self.lam = lam
        self.n_splines = n_splines
        self.gam_ = None

    def _to_dense_float(self, X):
        # Handle sparse matrices
        if sparse.issparse(X):
            return X.astype(np.float64).toarray()
        # Handle pandas DataFrame
        if hasattr(X, "to_numpy"):
            return X.to_numpy(dtype=np.float64)
        # Fallback: numpy array / list
        return np.asarray(X, dtype=np.float64)

    def fit(self, X, y):
        X_dense = self._to_dense_float(X)
        self.gam_ = LogisticGAM(lam=self.lam, n_splines=self.n_splines)
        self.gam_.fit(X_dense, y)
        return self

    def predict_proba(self, X):
        X_dense = self._to_dense_float(X)
        p1 = self.gam_.predict_proba(X_dense)  # P(y=1)
        p1 = np.asarray(p1, dtype=np.float64)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)




class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, weights):
        self.base_models = base_models
        self.weights = weights

    def fit(self, X, y):
        self.fitted_models_ = {}
        # normalize weights to be safe
        total_w = sum(self.weights.values())
        if total_w <= 0:
            raise ValueError("Total ensemble weight must be > 0.")
        self.norm_weights_ = {k: v / total_w for k, v in self.weights.items()}

        for name, model in self.base_models.items():
            if self.norm_weights_.get(name, 0.0) <= 0.0:
                continue
            self.fitted_models_[name] = clone(model).fit(X, y)

        return self

    def predict_proba(self, X):
        if not hasattr(self, "fitted_models_"):
            raise RuntimeError("Call fit before predict_proba.")

        p_ens = None
        for name, model in self.fitted_models_.items():
            w = self.norm_weights_[name]
            proba = model.predict_proba(X)  # shape (n_samples, 2)
            p1 = proba[:, 1]
            if p_ens is None:
                p_ens = np.zeros_like(p1, dtype=float)
            p_ens += w * p1

        p_ens = np.clip(p_ens, 1e-6, 1 - 1e-6)
        p0 = 1.0 - p_ens
        return np.column_stack([p0, p_ens])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )


def build_base_models():
    pre = build_preprocessor()

    # Bagging (over decision trees) 
    bagging_base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )
    bagging = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                BaggingClassifier(
                    estimator=bagging_base,
                    n_estimators=600,
                    max_samples=0.9,
                    max_features=0.8,
                    random_state=42,
                ),
            ),
        ]
    )

    # LightGBM
    lgbm = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LGBMClassifier(
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
                ),
            ),
        ]
    )

    # GAM 
    gam = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", GamWrapper(lam=1000.0, n_splines=8)),
        ]
    )

    # Logistic Regression – C 
    lr = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    C=0.00126743,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=500,
                ),
            ),
        ]
    )

    return {
        "bagging": bagging,
        "lgbm": lgbm,
        "gam": gam,
        "lr": lr,
    }


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # Basic sanity check
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in CSV: {missing_cols}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    print(f"Dataset shape after dropna: X={X.shape}, y={y.shape}")

    base_models = build_base_models()
    ensemble = WeightedEnsemble(base_models=base_models, weights=ENSEMBLE_WEIGHTS)

    print("Fitting weighted ensemble on full dataset ...")
    ensemble.fit(X, y)

    print(f"Saving trained ensemble to {OUTPUT_PKL} ...")
    joblib.dump(ensemble, OUTPUT_PKL)

    print("Done.")


if __name__ == "__main__":
    main()
