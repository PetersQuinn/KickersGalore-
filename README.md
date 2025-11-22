Field Goal Probability Modeling – NFL Kicker Analytics

A calibrated machine-learning system for estimating field-goal make probability in real NFL conditions

📌 Project Overview

NFL field goals are high-stakes, data-rich events—but the real challenge isn't predicting make vs miss.
It’s estimating probability of success under dynamic conditions like distance, wind, surface, pressure, and kicker history.

This project builds a full end-to-end predictive pipeline that:

Loads and cleans 12,449 NFL field-goal attempts (2013–2024)

Engineered features across weather, stadium, game context, and kicker career trends

Trains multiple model families (linear, tree-based, Bayesian, additive models)

Applies per-distance isotonic calibration for robust probability estimates

Combines the best models in a weighted ensemble

Evaluates performance using Brier Score, AUC, ECE, and PR-AUC

Demonstrates results via a live in-class experiment analyzing three real kicks

The result: a calibrated, interpretable, and football-specific probability model designed to support real decision-making—not just accuracy numbers.

🎯 Goal of the Project

Build a reliable model for in-game decision support.

Instead of “will the kicker make it?”, coaches need:

“What is the probability we make this kick—right here, right now?”

This enables smarter choices between:

Attempting the field goal

Going for it on 4th down

Punting

Managing the clock differently

The entire modeling pipeline prioritizes probabilistic calibration, risk awareness, and real-world interpretability.
