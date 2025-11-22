import streamlit as st
import pandas as pd
from pathlib import Path

import numpy as np
from scipy import sparse
from pygam import LogisticGAM

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


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

TARGET_COL = "field_goal_result_binary"
CATEGORICAL_COLS = ["kicker_player_name"]

# Best weighted ensemble from your search:
ENSEMBLE_WEIGHTS = {
    "bagging": 0.10,
    "lgbm": 0.35,
    "gam": 0.40,
    "lr": 0.15,
}

DATA_PATH = Path(__file__).resolve().parent / "data_for_streamlit.csv"


def build_preprocessor():
    numeric_cols = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", numeric_cols),
        ]
    )


def build_base_models():
    pre = build_preprocessor()

    # Bagging
    bagging_base = DecisionTreeClassifier(
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
    )
    bagging = Pipeline(
        steps=[
            ("pre", pre),
            ("clf",
             BaggingClassifier(
                 estimator=bagging_base,
                 n_estimators=600,
                 max_samples=0.9,
                 max_features=0.8,
                 random_state=42,
             )
            ),
        ]
    )

    # LightGBM
    lgbm = Pipeline(
        steps=[
            ("pre", pre),
            ("clf",
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
             )
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

    # Logistic Regression
    lr = Pipeline(
        steps=[
            ("pre", pre),
            ("clf",
             LogisticRegression(
                 C=0.00126743,
                 penalty="l2",
                 solver="lbfgs",
                 max_iter=500,
             )
            ),
        ]
    )

    return {
        "bagging": bagging,
        "lgbm": lgbm,
        "gam": gam,
        "lr": lr,
    }


@st.cache_resource
def load_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Sanity check columns
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    base_models = build_base_models()
    ensemble = WeightedEnsemble(base_models=base_models, weights=ENSEMBLE_WEIGHTS)
    ensemble.fit(X, y)

    return ensemble


def safe_load_model():
    try:
        return load_model(), None
    except Exception as e:
        return None, str(e)

def build_input_form():
    st.header("Kick Scenario")

    with st.expander("Game context", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            season = st.number_input("Season", min_value=1990, max_value=2035,
                                     value=2024, step=1)
            score_diff = st.number_input(
                "Score differential (offense - defense)",
                min_value=-40,
                max_value=40,
                value=-3,
                step=1,
                help="Negative if your team is losing, positive if winning.",
            )

        with c2:
            quarter_4 = st.checkbox("4th quarter?", value=True)
            buzzer_beater = st.checkbox(
                "Buzzer-beater (last-second) attempt?",
                value=False,
                help="Clock near 0, game-deciding kick.",
            )

        with c3:
            season_type = st.selectbox(
                "Game type",
                options=["Regular season", "Postseason"],
                index=0,
            )

    with st.expander("Environment", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            distance = st.number_input(
                "Kick distance (yards)", min_value=15, max_value=70,
                value=45, step=1
            )
            altitude = st.number_input(
                "Stadium altitude (feet)",
                min_value=0,
                max_value=8000,
                value=0,
                step=50,
                help="Approx stadium altitude; 0 if unknown.",
            )

        with c2:
            temp = st.number_input(
                "Temperature (°F)", min_value=-10, max_value=120,
                value=50, step=1
            )
            wind = st.number_input(
                "Wind speed (mph)", min_value=0, max_value=40,
                value=5, step=1
            )

        with c3:
            roof = st.selectbox(
                "Roof",
                options=["Open", "Closed"],
                index=0,
            )
            surface = st.selectbox(
                "Surface",
                options=["Grass", "Turf"],
                index=0,
            )

        c4, c5 = st.columns(2)
        with c4:
            is_rain = st.checkbox("Raining?", value=False)
            is_snow = st.checkbox("Snowing?", value=False)
        with c5:
            vegas_wp = st.slider(
                "Vegas win probability (offense before the kick)",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.01,
                help="If you don't have this, leave at 0.50.",
            )

    with st.expander("Kicker profile", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            kicker_name = st.text_input(
                "Kicker name",
                value="Kicker Name",
                help="Used as a categorical feature.",
            )

        with c2:
            career_attempts = st.number_input(
                "Career FG attempts (before this kick)",
                min_value=0,
                max_value=1000,
                value=250,
                step=1,
            )

        with c3:
            career_fg_pct = st.slider(
                "Career FG% (0 - 1.0)",
                min_value=0.0,
                max_value=1.0,
                value=0.88,
                step=0.01,
            )

    # Binary encodings
    season_type_binary = 1 if season_type == "Postseason" else 0
    roof_binary = 1 if roof == "Closed" else 0
    surface_binary = 1 if surface == "Turf" else 0
    is_4th_qtr = 1 if quarter_4 else 0
    buzzer_beater_binary = 1 if buzzer_beater else 0

    # Build 1-row DataFrame in the exact column order
    row = pd.DataFrame(
        [{
            "season": int(season),
            "score_differential": float(score_diff),
            "kicker_player_name": kicker_name,
            "kick_distance": float(distance),
            "temp": float(temp),
            "wind": float(wind),
            "season_type_binary": int(season_type_binary),
            # NOTE: target field_goal_result_binary is NOT included here
            "roof_binary": int(roof_binary),
            "surface_binary": int(surface_binary),
            "altitude": float(altitude),
            "vegas_wp_effective": float(vegas_wp),
            "is_rain": int(is_rain),
            "is_snow": int(is_snow),
            "career_attempts": int(career_attempts),
            "career_fg_pct": float(career_fg_pct),
            "is_4th_qtr": int(is_4th_qtr),
            "buzzer_beater_binary": int(buzzer_beater_binary),
        }],
        columns=FEATURE_COLS,
    )

    return row


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Kickers Galore – Field Goal Probability",
        page_icon="🏈",
        layout="wide",
    )

    st.title("🏈 Kickers Galore – Field Goal Make Probability")
    st.markdown(
        """
        This app uses Quinton Peters' **weighted ensemble model** to estimate the chance a field goal is made,
        given game, environment, and kicker context.

        This model was fine tuned on over 10,000 NFL kicks and combines multiple machine learning algorithms
        (LightGBM, GAM, Bagging, Logistic Regression) to provide robust predictions.

        Please be patient when loading the model; it should take several minutes.
        """
    )

    model, model_err = safe_load_model()
    if model is None:
        st.error(
            "Model not loaded.\n\n"
            f"Details: `{model_err}`\n\n"
            "Make sure `field_goals_model_ready.csv` exists next to this app, "
            "then refresh the page."
        )
        st.stop()

    input_df = build_input_form()

    st.subheader("Model Input Preview")
    st.dataframe(input_df)

    if st.button("Compute Make / Miss Probability", type="primary"):
        try:
            proba = model.predict_proba(input_df)[0]
            # Assuming class order [0=miss, 1=make]
            p_miss = float(proba[0])
            p_make = float(proba[1])

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Estimated Make Probability", f"{p_make*100:.1f}%")
            with c2:
                st.metric("Estimated Miss Probability", f"{p_miss*100:.1f}%")

            st.progress(min(max(p_make, 0.0), 1.0))

            st.markdown(
                f"""
                **Interpretation**

                Given a **{int(input_df['kick_distance'].iloc[0])}-yard** attempt by 
                **{input_df['kicker_player_name'].iloc[0]}** with the selected conditions, 
                this model estimates roughly a **{p_make*100:.1f}%** chance of a make.

                Remember: this is a **probabilistic** model, not a guarantee. 
                Edge cases (extreme weather, injuries, botched snaps) may behave very differently.
                This model does NOT account for blocked kicks or other unusual events.
                """
            )
        except Exception as e:
            st.error(f"Error when running prediction: `{e}`")


if __name__ == "__main__":
    main()
