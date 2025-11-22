"""
EDA for field_goals_model_ready.csv (Kicker Project)

Run:
    python eda_kicker_project.py

Assumes columns:
season,score_differential,kicker_player_name,kick_distance,temp,wind,
season_type_binary,field_goal_result_binary,roof_binary,surface_binary,
altitude,vegas_wp_effective,is_rain,is_snow,career_attempts,career_fg_pct,
is_4th_qtr,buzzer_beater_binary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CSV_PATH = "field_goals_model_ready.csv"

df = pd.read_csv(CSV_PATH)

print("\n=== BASIC DATASET INFO ===")
print(f"Rows (field goal attempts): {len(df):,}")
print("\nColumns:")
print(df.columns.tolist())

print("\nDtypes:")
print(df.dtypes)

print("\nHead:")
print(df.head())

# Helper: safe numeric conversion
def to_num(series):
    return pd.to_numeric(series, errors="coerce")


# =========================
# CORE OUTCOME + DISTANCE
# =========================

print("\n=== CORE OUTCOME & DISTANCE STATS ===")

# Overall make percentage
if "field_goal_result_binary" in df.columns:
    outcome = to_num(df["field_goal_result_binary"])
    make_rate = outcome.mean()
    print(f"\nOverall Make Percentage (field_goal_result_binary): "
          f"{make_rate:.3f}  ({make_rate * 100:.1f}%)")
else:
    print("\n[WARN] field_goal_result_binary not found.")

# Distance stats + buckets
if "kick_distance" in df.columns:
    dist = to_num(df["kick_distance"])
    print("\nKick Distance Stats (kick_distance):")
    print(f"  Mean distance: {dist.mean():.2f} yards")
    print(f"  Std dev distance: {dist.std():.2f} yards")
    print(f"  Min distance: {dist.min():.0f} yards")
    print(f"  Max distance: {dist.max():.0f} yards")

    # =========================
    # Histogram of kick distance
    # =========================
    plt.figure(figsize=(8, 5))
    plt.hist(dist, bins=range(10, 80, 1))  # 5-yard bins from 10 to 75
    plt.xlabel("Kick Distance (yards)")
    plt.ylabel("Number of Attempts")
    plt.title("Distribution of NFL Field Goal Distances")
    plt.tight_layout()
    plt.savefig("kick_distance_histogram.png", dpi=300)
    plt.show()  # uncomment if you want an interactive window


    # Distance buckets
    bins = [0, 29, 39, 49, 100]
    labels = ["0–29", "30–39", "40–49", "50+"]
    dist_bucket = pd.cut(dist, bins=bins, labels=labels, right=True)
    bucket_counts = dist_bucket.value_counts().sort_index()
    bucket_perc = (bucket_counts / len(df)) * 100

    print("\nDistance Bucket Distribution:")
    for label in labels:
        count = bucket_counts.get(label, 0)
        perc = bucket_perc.get(label, 0.0)
        print(f"  {label} yards: {count:>5} kicks ({perc:5.1f}%)")

    # Optional: make % by distance bucket
    if "field_goal_result_binary" in df.columns:
        bucket_make = (
            df.groupby(dist_bucket)["field_goal_result_binary"]
              .mean()
              .reindex(labels)
        )
        print("\nMake Percentage by Distance Bucket:")
        for label in labels:
            val = bucket_make.get(label, np.nan)
            if pd.isna(val):
                continue
            print(f"  {label} yards: {val * 100:5.1f}%")
else:
    print("\n[WARN] kick_distance not found.")


# =========================
# CONTEXT & GAME SITUATION
# =========================

print("\n=== CONTEXT & GAME SITUATION ===")

# Season type (binary)
if "season_type_binary" in df.columns:
    s = to_num(df["season_type_binary"])
    counts = s.value_counts(dropna=False)
    perc = counts / len(df) * 100
    print("\nSeason Type (season_type_binary):")
    for val, count in counts.items():
        print(f"  Value {val}: {count:>5} kicks ({perc[val]:5.1f}%)")
    print("  -> Interpret value '1' as playoffs if that's your encoding.")
else:
    print("\n[INFO] season_type_binary not found.")

# 4th quarter
if "is_4th_qtr" in df.columns:
    q4 = to_num(df["is_4th_qtr"])
    rate_q4 = q4.mean()
    print("\nFourth Quarter Kicks (is_4th_qtr):")
    print(f"  Number in 4th quarter: {int(q4.sum()):,}")
    print(f"  Percentage of all kicks in 4th: {rate_q4 * 100:.2f}%")
else:
    print("\n[INFO] is_4th_qtr not found.")

# Buzzer beater / high pressure
if "buzzer_beater_binary" in df.columns:
    bb = to_num(df["buzzer_beater_binary"])
    rate_bb = bb.mean()
    print("\nHigh-Pressure / Buzzer-Beater Kicks (buzzer_beater_binary):")
    print(f"  Number of high-pressure kicks: {int(bb.sum()):,}")
    print(f"  Percentage of all kicks high-pressure: {rate_bb * 100:.2f}%")
else:
    print("\n[INFO] buzzer_beater_binary not found.")

# Score differential
if "score_differential" in df.columns:
    sd = to_num(df["score_differential"])
    print("\nScore Differential (offense score - defense score) stats:")
    print(f"  Mean: {sd.mean():.2f}")
    print(f"  Std dev: {sd.std():.2f}")
    print(f"  Min: {sd.min():.0f}")
    print(f"  Max: {sd.max():.0f}")
else:
    print("\n[INFO] score_differential not found.")


# =========================
# STADIUM / WEATHER
# =========================

print("\n=== STADIUM & WEATHER ===")

# Roof
if "roof_binary" in df.columns:
    roof_counts = df["roof_binary"].value_counts(dropna=False)
    roof_perc = roof_counts / len(df) * 100
    print("\nRoof Type (roof_binary):")
    for val, count in roof_counts.items():
        print(f"  {val}: {count:>5} kicks ({roof_perc[val]:5.1f}%)")
else:
    print("\n[INFO] roof_binary not found.")

# Surface
if "surface_binary" in df.columns:
    surf_counts = df["surface_binary"].value_counts(dropna=False)
    surf_perc = surf_counts / len(df) * 100
    print("\nSurface Type (surface_binary):")
    for val, count in surf_counts.items():
        print(f"  {val}: {count:>5} kicks ({surf_perc[val]:5.1f}%)")
else:
    print("\n[INFO] surface_binary not found.")

# Temperature
if "temp" in df.columns:
    temp = to_num(df["temp"])
    print("\nTemperature (temp) stats:")
    print(f"  Mean: {temp.mean():.1f}°F")
    print(f"  Std dev: {temp.std():.1f}°F")
    print(f"  Min: {temp.min():.1f}°F")
    print(f"  Max: {temp.max():.1f}°F")
else:
    print("\n[INFO] temp not found.")

# Wind
if "wind" in df.columns:
    wind = to_num(df["wind"])
    print("\nWind (wind) stats:")
    print(f"  Mean: {wind.mean():.1f} mph")
    print(f"  Std dev: {wind.std():.1f} mph")
    print(f"  Min: {wind.min():.1f} mph")
    print(f"  Max: {wind.max():.1f} mph")
else:
    print("\n[INFO] wind not found.")

# Rain / Snow
if "is_rain" in df.columns:
    r = to_num(df["is_rain"])
    rate_r = r.mean()
    print("\nRain Games (is_rain):")
    print(f"  Kicks in rain: {int(r.sum()):,}")
    print(f"  Percentage in rain: {rate_r * 100:.2f}%")
else:
    print("\n[INFO] is_rain not found.")

if "is_snow" in df.columns:
    s = to_num(df["is_snow"])
    rate_s = s.mean()
    print("\nSnow Games (is_snow):")
    print(f"  Kicks in snow: {int(s.sum()):,}")
    print(f"  Percentage in snow: {rate_s * 100:.2f}%")
else:
    print("\n[INFO] is_snow not found.")

# Altitude
if "altitude" in df.columns:
    alt = to_num(df["altitude"])
    print("\nAltitude stats (altitude, in feet):")
    print(f"  Mean: {alt.mean():.1f}")
    print(f"  Std dev: {alt.std():.1f}")
    print(f"  Min: {alt.min():.1f}")
    print(f"  Max: {alt.max():.1f}")
else:
    print("\n[INFO] altitude not found.")


# =========================
# KICKER & WIN PROBABILITY
# =========================

print("\n=== KICKER HISTORY & WIN PROBABILITY ===")

# Vegas win probability
if "vegas_wp_effective" in df.columns:
    wp = to_num(df["vegas_wp_effective"])
    print("\nVegas Win Probability (vegas_wp_effective):")
    print(f"  Mean: {wp.mean():.3f}")
    print(f"  Std dev: {wp.std():.3f}")
    print(f"  Min: {wp.min():.3f}")
    print(f"  Max: {wp.max():.3f}")
else:
    print("\n[INFO] vegas_wp_effective not found.")

# Career attempts
if "career_attempts" in df.columns:
    ca = to_num(df["career_attempts"])
    print("\nCareer Attempts (career_attempts) stats:")
    print(f"  Mean: {ca.mean():.1f}")
    print(f"  Std dev: {ca.std():.1f}")
    print(f"  Min: {ca.min():.0f}")
    print(f"  Max: {ca.max():.0f}")
else:
    print("\n[INFO] career_attempts not found.")

# Career FG %
if "career_fg_pct" in df.columns:
    cfp = to_num(df["career_fg_pct"])
    print("\nCareer Field Goal Percentage (career_fg_pct):")
    print(f"  Mean: {cfp.mean():.3f}  ({cfp.mean() * 100:.1f}%)")
    print(f"  Std dev: {cfp.std():.3f}")
    print(f"  Min: {cfp.min():.3f}")
    print(f"  Max: {cfp.max():.3f}")
else:
    print("\n[INFO] career_fg_pct not found.")

# Number of unique kickers
if "kicker_player_name" in df.columns:
    num_kickers = df["kicker_player_name"].nunique()
    print(f"\nUnique kickers (kicker_player_name): {num_kickers}")
else:
    print("\n[INFO] kicker_player_name not found.")

print("\n=== DONE: Use these stats to build your EDA slides. ===")
