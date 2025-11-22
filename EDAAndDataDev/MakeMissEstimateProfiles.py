# Quick & dirty "bell curve" visuals from summary stats only.
# For each model, we approximate:
#   P(make | true miss) ~ Normal(mean_miss, std_miss)
#   P(make | true make) ~ Normal(mean_make, std_make)
# truncated to [0, 1].
#
# This is a loose visual (not using the raw probabilities)

import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "bell_curves_from_summaries"
# All are for P(make) on TEST

MODEL_STATS = {
    "Bagging": {
        "miss_mean": 0.7672,
        "miss_std": 0.1050,
        "make_mean": 0.8735,
        "make_std": 0.1120,
    },
    "Bayes-LR": {
        "miss_mean": 0.7175,
        "miss_std": 0.1525,
        "make_mean": 0.8559,
        "make_std": 0.1369,
    },
    "LightGBM": {
        "miss_mean": 0.7357,
        "miss_std": 0.1192,
        "make_mean": 0.8561,
        "make_std": 0.1245,
    },
    "GAM": {
        "miss_mean": 0.7577,
        "miss_std": 0.1253,
        "make_mean": 0.8755,
        "make_std": 0.1203,
    },
    "BART": {
        "miss_mean": 0.7473,
        "miss_std": 0.1432,
        "make_mean": 0.8717,
        "make_std": 0.1211,
    },
    "Logistic Regression": {
        "miss_mean": 0.7229,
        "miss_std": 0.1510,
        "make_mean": 0.8596,
        "make_std": 0.1352,
    },
}

def normal_pdf(x, mean, std):
    std = max(std, 1e-6)  # defensive
    return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # X-axis from 0 to 1 (probability)
    x = np.linspace(0.0, 1.0, 500)

    for model_name, stats in MODEL_STATS.items():
        miss_mu = stats["miss_mean"]
        miss_sd = stats["miss_std"]
        make_mu = stats["make_mean"]
        make_sd = stats["make_std"]

        # Compute pdfs on [0,1]
        y_miss = normal_pdf(x, miss_mu, miss_sd)
        y_make = normal_pdf(x, make_mu, make_sd)

        max_y = max(y_miss.max(), y_make.max())
        if max_y > 0:
            y_miss = y_miss / max_y
            y_make = y_make / max_y

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y_miss, label="True MISSES", linestyle="--")
        ax.plot(x, y_make, label="True MAKES", linestyle="-")

        ax.set_title(f"{model_name}: Approx. bell curves of P(make)")
        ax.set_xlabel("Predicted probability of MAKE")
        ax.set_ylabel("Relative density (normalized)")
        ax.set_xlim(0.0, 1.0)
        ax.legend()
        ax.grid(alpha=0.2)

        plt.tight_layout()
        out_path = os.path.join(
            OUT_DIR,
            model_name.lower().replace(" ", "_").replace("-", "_") + "_bell.png",
        )
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[PLOT] Saved {model_name} bell curve to: {out_path}")

if __name__ == "__main__":
    main()
