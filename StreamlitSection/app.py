import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

import numpy as np
from scipy import sparse
from pygam import LogisticGAM
from sklearn.base import BaseEstimator, ClassifierMixin

# =========================================================
# Classes needed to unpickle the trained ensemble
# =========================================================

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
    """
    Simple weighted average of base model probabilities.

    base_models: dict[name -> estimator] (each must support fit / predict_proba)
    weights: dict[name -> float], should sum to 1.0 (we normalize just in case).
    """

    def __init__(self, base_models, weights):
        self.base_models = base_models
        self.weights = weights

    def fit(self, X, y):
        # Not used in the app (we only load a pre-fit model),
        # but kept for compatibility with the object that was pickled.
        from sklearn.base import clone

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
            # In your loaded model, this will already exist from training.
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


# =========================================================
# Original app config
# =========================================================

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

MODEL_FILENAME = "weighted_ensemble.pkl"


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Run your FinalModelDownloader script to create {MODEL_FILENAME}."
        )
    return joblib.load(model_path)


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
            season = st.number_input("Season", min_value=1990, max_value=2035, value=2024, step=1)
            score_diff = st.number_input(
                "Score differential (offense - defense)",
                min_value=-40,
                max_value=40,
                value=-3,
                step=1,
                help="Negative if your team is losing, positive if winning."
            )

        with c2:
            quarter_4 = st.checkbox("4th quarter?", value=True)
            buzzer_beater = st.checkbox(
                "Buzzer-beater (last-second) attempt?",
                value=False,
                help="Clock near 0, game-deciding kick."
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
                "Kick distance (yards)", min_value=15, max_value=70, value=45, step=1
            )
            altitude = st.number_input(
                "Stadium altitude (feet)",
                min_value=0,
                max_value=8000,
                value=0,
                step=50,
                help="Approx stadium altitude; 0 if unknown."
            )

        with c2:
            temp = st.number_input(
                "Temperature (°F)", min_value=-10, max_value=120, value=50, step=1
            )
            wind = st.number_input(
                "Wind speed (mph)", min_value=0, max_value=40, value=5, step=1
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
                help="If you don't have this, leave at 0.50."
            )

    with st.expander("Kicker profile", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            kicker_name = st.text_input(
                "Kicker name",
                value="Justin Tucker",
                help="Used as a categorical feature."
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
    row = pd.DataFrame([{
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
    }], columns=FEATURE_COLS)

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
        This app uses your **weighted ensemble model** to estimate the chance a field goal is made,
        given game situation, environment, and kicker profile.
        """
    )

    model, model_err = safe_load_model()
    if model is None:
        st.error(
            "Model not loaded.\n\n"
            f"Details: `{model_err}`\n\n"
            f"Make sure `{MODEL_FILENAME}` exists in the same folder as this app, "
            "then rerun the Streamlit app."
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
                """
            )
        except Exception as e:
            st.error(f"Error when running prediction: `{e}`")


if __name__ == "__main__":
    main()
