# kicker_ensemble_eval.py
#
# Load per-model probability CSVs from model_probs/ and:
#   - Evaluate each individual model.
#   - Try all equal-weight average ensembles across models (size >= 2).
#   - Print metrics in a similar format to the single-model scripts.
#
# Metrics:
#   Brier (on P(make)), AUC, PR-AUC(miss), ECE@10, confusion @ 0.5.
#   For the best ensemble, also print probability diagnostics.

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

N_ECE_BINS = 10
PROBS_DIR = Path("model_probs")

# Map short model names to CSV filenames produced by the previous script.
MODEL_FILES = {
    "bagging": "probs_bagging.csv",
    "bayes_lr": "probs_bayes_lr.csv",
    "lgbm": "probs_lgbm.csv",
    "gam": "probs_gam.csv",
    "bart": "probs_bart.csv",
    "lr": "probs_lr.csv",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def ece_score(probs, y, n_bins=N_ECE_BINS):
    """
    Expected Calibration Error.
    probs: P(make), y: 1=make, 0=miss
    """
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


def summarize_probs(name, arr):
    arr = np.asarray(arr)
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


def compute_metrics(p_make, y_true, name=""):
    """
    Compute all the scalar metrics we care about for a given P(make).
    """
    y_true = np.asarray(y_true)
    p_make = np.asarray(p_make)

    # Primary: Brier on P(make)
    brier = brier_score_loss(y_true, p_make)

    # AUC on P(make) vs y (1=make)
    auc = roc_auc_score(y_true, p_make)

    # PR-AUC for misses: treat misses as positives, use P(miss) = 1 - P(make)
    y_miss = 1 - y_true
    p_miss = 1.0 - p_make
    pr_miss = average_precision_score(y_miss, p_miss)

    # ECE@10 on P(make)
    ece10 = ece_score(p_make, y_true, n_bins=N_ECE_BINS)

    # Confusion matrix at threshold 0.5 on P(make)
    thr = 0.5
    yhat = (p_make >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()

    metrics = {
        "name": name,
        "brier": brier,
        "auc": auc,
        "pr_miss": pr_miss,
        "ece10": ece10,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "thr": thr,
    }
    return metrics


def print_metrics_block(m):
    """
    Pretty-print metrics like in the single-model scripts.
    """
    print(f"\n=== TEST — {m['name']} ===")
    print(f"Brier={m['brier']:.5f} (primary, on P(make))")
    print(
        f"AUC={m['auc']:.4f} | PR-AUC(miss)={m['pr_miss']:.4f} | "
        f"ECE@{N_ECE_BINS}={m['ece10']:.4f}"
    )
    print(f"Threshold for confusion matrix (on P(make)): {m['thr']:.2f}")
    print(
        f"Confusion matrix (1=make, 0=miss): "
        f"tn={m['tn']} fp={m['fp']} fn={m['fn']} tp={m['tp']}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # ---- Load per-model CSVs and assemble into a single frame ----
    frames = {}
    y_ref = None

    for mname, fname in MODEL_FILES.items():
        path = PROBS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file for model '{mname}': {path}")
        df = pd.read_csv(path)
        if "y_te" not in df.columns or "p_make" not in df.columns:
            raise ValueError(f"{path} must contain columns 'y_te' and 'p_make'.")

        if y_ref is None:
            y_ref = df["y_te"].values
        else:
            if not np.array_equal(y_ref, df["y_te"].values):
                raise ValueError(f"y_te mismatch in {path} compared to others.")

        frames[mname] = df["p_make"].values

    y_te = y_ref
    model_names = list(frames.keys())
    print("Loaded models:", ", ".join(model_names))
    print(f"Test size: {len(y_te)}")

    # Build a single DataFrame for convenience
    preds_df = pd.DataFrame({"y_te": y_te})
    for mname in model_names:
        preds_df[mname] = frames[mname]

    # ---- Evaluate each individual model (baseline) ----
    print("\n---------------- Individual models ----------------")
    indiv_metrics = []
    for mname in model_names:
        m = compute_metrics(preds_df[mname].values, y_te, name=mname)
        indiv_metrics.append(m)
        print_metrics_block(m)

    # ---- Try all equal-weight ensembles for subsets of size >= 2 ----
    print("\n---------------- Ensembles (equal-weight average) ----------------")

    ensemble_metrics = []

    for k in range(2, len(model_names) + 1):
        for combo in itertools.combinations(model_names, k):
            combo_name = "ENS[" + "+".join(combo) + "]"
            p_cols = [preds_df[m].values for m in combo]
            p_ens = np.mean(np.stack(p_cols, axis=0), axis=0)
            m = compute_metrics(p_ens, y_te, name=combo_name)
            ensemble_metrics.append(m)

    # Sort ensembles by Brier (lower is better)
    ensemble_metrics_sorted = sorted(ensemble_metrics, key=lambda x: x["brier"])

    # ---- Print top ensembles summary ----
    TOP_K = 10
    print(f"\nTop {TOP_K} ensembles by Brier score:")
    print(
        "rank | ensemble_name                                  "
        "| Brier    | AUC    | PR-miss | ECE@10"
    )
    print("-" * 90)
    for i, m in enumerate(ensemble_metrics_sorted[:TOP_K], start=1):
        print(
            f"{i:4d} | {m['name']:<45} "
            f"| {m['brier']:.5f} | {m['auc']:.4f} | {m['pr_miss']:.4f} | {m['ece10']:.4f}"
        )

    # ---- Detailed diagnostics for the best ensemble ----
    best_ens = ensemble_metrics_sorted[0]
    combo_str = best_ens["name"]
    print("\n==============================================================")
    print("Best ensemble by Brier:", combo_str)
    print_metrics_block(best_ens)

    # Reconstruct its prediction vector
    # name format is ENS[m1+m2+...], so extract the model names from inside brackets.
    inside = combo_str[len("ENS[") : -1]
    best_models = inside.split("+")
    p_cols = [preds_df[m].values for m in best_models]
    p_best = np.mean(np.stack(p_cols, axis=0), axis=0)

    # Probability diagnostics like before
    y_te_arr = np.asarray(y_te)
    p_make_miss = p_best[y_te_arr == 0]
    p_make_make = p_best[y_te_arr == 1]

    print("\n--- Probability Diagnostics on TEST (Best Ensemble) ---")
    summarize_probs("True MISSES (y=0) – P(make)", p_make_miss)
    summarize_probs("True MAKES (y=1) – P(make)", p_make_make)

    for t in [0.60, 0.70, 0.80, 0.90]:
        frac = np.mean(p_make_miss < t)
        print(f"Fraction of MISSES with P(make) < {t:.2f}: {frac:.3f}")


if __name__ == "__main__":
    main()
