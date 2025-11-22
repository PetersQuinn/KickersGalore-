# kicker_weighted_ensembles.py
# Full weighted ensemble search over 6 tuned, calibrated models.
#
# Assumes you already saved per-model TEST predictions as CSVs with:
#   - a common "fg_id" or index identifying each kick (we'll use index)
#   - column "y_te"  : 1=make, 0=miss
#   - column "p_make"  : calibrated P(make) for that model
#
# Expected filenames (in MODEL_PRED_DIR):
#   pred_bagging.csv
#   pred_bayes_lr.csv
#   pred_lgbm.csv
#   pred_gam.csv
#   pred_bart.csv
#   pred_lr.csv
#
# Output:
#   - Prints top K ensembles by Brier
#   - Prints full diagnostics for best ensemble.

import os
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

# ---------------- config ----------------

MODEL_PRED_DIR = Path("model_probs")  # folder where your per-model CSVs live
GRID_STEP = 0.05                      # weight step; 0.05 -> weights in {0,0.05,...,1.0}
TOP_K = 20                            # how many best ensembles to print
N_ECE_BINS = 10                       # for calibration

# Model names and corresponding CSV filenames
MODEL_SPECS = {
    "bagging":   "probs_bagging.csv",
    "bayes_lr":  "probs_bayes_lr.csv",
    "lgbm":      "probs_lgbm.csv",
    "gam":       "probs_gam.csv",
    "bart":      "probs_bart.csv",
    "lr":        "probs_lr.csv",
}


# ---------------- helpers ----------------

def ece_score(probs, y, n_bins=N_ECE_BINS):
    """
    Expected Calibration Error (ECE).
    probs: P(make), y: 1=make, 0=miss
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)

    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    N = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        e += m.sum() / N * abs(probs[m].mean() - y[m].mean())
    return float(e)


def load_all_model_preds():
    """
    Load all per-model prediction CSVs and merge into a single DataFrame.
    Assumes each file has columns ['y_te', 'p_make'] and identical row order.
    """
    data = None
    model_cols = {}

    for name, fname in MODEL_SPECS.items():
        path = MODEL_PRED_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {name}: {path}")

        df = pd.read_csv(path)
        # Expect y_te and p_make; keep them
        if "y_te" not in df.columns or "p_make" not in df.columns:
            raise ValueError(f"{path} must contain 'y_te' and 'p_make' columns.")

        if data is None:
            data = pd.DataFrame({
                "y_te": df["y_te"].astype(int).values
            })
        else:
            # sanity check: y_te must match
            if not np.array_equal(data["y_te"].values, df["y_te"].values):
                raise ValueError(f"y_te mismatch between files; check {fname}")

        col_name = f"p_make_{name}"
        data[col_name] = df["p_make"].astype(float).values
        model_cols[name] = col_name

    return data, model_cols


def generate_weight_tuples(n_models, grid_step=GRID_STEP):
    """
    Generate all non-negative weight vectors of length n_models with
    components in {0, grid_step, 2*grid_step, ..., 1.0} that sum to 1.0.
    Uses integer compositions: let TOTAL = 1 / grid_step, each weight = k_i / TOTAL.
    """
    total = int(round(1.0 / grid_step))
    # We produce all integer tuples (k0,...,k_{n-1}) with sum=k_total
    # Here n_models = 6, but we keep it general.
    if n_models == 1:
        yield (1.0,)
        return

    # recursive / iterative composition generator
    # We'll do an iterative nested loop with pruning using indices.
    def rec(prefix, remaining, slots_left):
        if slots_left == 1:
            # last slot gets whatever is left
            yield prefix + (remaining,)
            return
        # For current slot, can take 0..remaining
        for k in range(remaining + 1):
            yield from rec(prefix + (k,), remaining - k, slots_left - 1)

    for ks in rec((), total, n_models):
        weights = tuple(k / total for k in ks)
        yield weights


def evaluate_ensemble(y_te, P, weights, model_order):
    """
    y_te: array of shape (N,) with 1=make, 0=miss
    P: dict model_name -> array of P(make) shape (N,)
    weights: tuple of floats, same order as model_order
    """
    # build ensemble probs
    p_ens = np.zeros_like(y_te, dtype=float)
    for w, name in zip(weights, model_order):
        if w == 0.0:
            continue
        p_ens += w * P[name]

    # metrics
    # Brier on P(make)
    brier = brier_score_loss(y_te, p_ens)
    auc = roc_auc_score(y_te, p_ens)

    # PR-AUC for misses: flip labels/probs
    y_miss = 1 - y_te
    p_miss = 1 - p_ens
    pr_miss = average_precision_score(y_miss, p_miss)

    # calibration
    ece10 = ece_score(p_ens, y_te, n_bins=N_ECE_BINS)

    # confusion at 0.5
    thr = 0.5
    yhat = (p_ens >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()

    return {
        "brier": brier,
        "auc": auc,
        "pr_miss": pr_miss,
        "ece10": ece10,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "p_ens": p_ens,
    }


# ---------------- main ----------------

def main():
    print("Loading per-model predictions...")
    df, model_cols = load_all_model_preds()
    y_te = df["y_te"].values

    # Collect model probs into a dict for fast access
    model_order = ["bagging", "bayes_lr", "lgbm", "gam", "bart", "lr"]
    for m in model_order:
        if m not in model_cols:
            raise ValueError(f"Model {m} not found in loaded predictions.")
    P = {m: df[model_cols[m]].values for m in model_order}

    print("Starting weighted ensemble search...")
    print(f"Models: {model_order}")
    print(f"GRID_STEP = {GRID_STEP} (weights in {{0, {GRID_STEP}, ..., 1.0}}; sum=1)")

    results = []
    n_models = len(model_order)

    for weights in generate_weight_tuples(n_models, GRID_STEP):
        # optionally skip degenerate "single-model" ensembles if you want
        # but for now we keep them; they help sanity-check that the search
        # can recover single-model performance when weight=1 on one model.
        metrics = evaluate_ensemble(y_te, P, weights, model_order)
        # store summary, but keep p_ens only for the best later
        results.append({
            "weights": weights,
            "ensemble_name": "ENS[" + "+".join(
                f"{w:.2f}*{m}" for w, m in zip(weights, model_order) if w > 0
            ) + "]",
            "brier": metrics["brier"],
            "auc": metrics["auc"],
            "pr_miss": metrics["pr_miss"],
            "ece10": metrics["ece10"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        })

    # sort by Brier
    results_sorted = sorted(results, key=lambda r: r["brier"])
    print("\n---------------- Weighted Ensembles (grid search) ----------------\n")
    print(f"Top {TOP_K} ensembles by Brier score:")
    print("rank | ensemble_name                                          | Brier    | AUC    | PR-miss | ECE@10")
    print("-----------------------------------------------------------------------------------------------------")
    for i, r in enumerate(results_sorted[:TOP_K], start=1):
        print(
            f"{i:4d} | {r['ensemble_name'][:52]:52s} | "
            f"{r['brier']:.5f} | {r['auc']:.4f} | {r['pr_miss']:.4f} | {r['ece10']:.4f}"
        )

    # Best ensemble diagnostics
    best = results_sorted[0]
    best_weights = best["weights"]
    best_metrics = evaluate_ensemble(y_te, P, best_weights, model_order)
    p_ens = best_metrics["p_ens"]

    print("\n==============================================================")
    print("Best weighted ensemble by Brier:")
    print(f"{best['ensemble_name']}")
    print(
        f"weights (in order {model_order}): "
        f"{', '.join(f'{w:.2f}' for w in best_weights)}"
    )

    brier = best_metrics["brier"]
    auc = best_metrics["auc"]
    pr_miss = best_metrics["pr_miss"]
    ece10 = best_metrics["ece10"]
    tn, fp, fn, tp = best_metrics["tn"], best_metrics["fp"], best_metrics["fn"], best_metrics["tp"]

    print("\n=== TEST — Best Weighted Ensemble ===")
    print(f"Brier={brier:.5f} (primary, on P(make))")
    print(f"AUC={auc:.4f} | PR-AUC(miss)={pr_miss:.4f} | ECE@10={ece10:.4f}")
    print("Threshold for confusion matrix (on P(make)): 0.50")
    print(f"Confusion matrix (1=make, 0=miss): tn={tn} fp={fp} fn={fn} tp={tp}")

    # Probability diagnostics similar to earlier scripts
    y_te_make = y_te  # 1=make, 0=miss
    p_make = p_ens

    p_make_miss = p_make[y_te_make == 0]  # predicted P(make) for true misses
    p_make_make = p_make[y_te_make == 1]  # predicted P(make) for true makes

    def summarize(name, arr):
        arr = np.asarray(arr, dtype=float)
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

    print("\n--- Probability Diagnostics on TEST (Best Weighted Ensemble) ---")
    summarize("True MISSES (y=0) – P(make)", p_make_miss)
    summarize("True MAKES (y=1) – P(make)", p_make_make)

    for t in [0.60, 0.70, 0.80, 0.90]:
        frac = np.mean(p_make_miss < t)
        print(f"Fraction of MISSES with P(make) < {t:.2f}: {frac:.3f}")


if __name__ == "__main__":
    main()
