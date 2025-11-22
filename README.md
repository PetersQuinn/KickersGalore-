# **Field Goal Probability Modeling – NFL Kicker Analytics**

*A calibrated machine-learning system for estimating field-goal make probability in real NFL conditions*

---

## **📌 Project Overview**

NFL field goals are high-stakes, data-rich events—but the real challenge isn't predicting *make vs miss*. It’s estimating **probability of success** under dynamic conditions like distance, wind, surface, pressure, and kicker history.

This project builds a full **end-to-end predictive pipeline** that:

* Loads and cleans 12,449 NFL field-goal attempts (2013–2024)
* Engineers features across **weather**, **stadium**, **game context**, and **kicker career trends**
* Trains multiple model families (linear, tree-based, Bayesian, additive models)
* Applies **per-distance isotonic calibration** for robust probability estimates
* Combines the best models in a **weighted ensemble**
* Evaluates performance using **Brier Score**, **AUC**, **ECE**, and **PR-AUC**
* Demonstrates results via a **live in-class experiment** analyzing three real kicks

The result: a calibrated, interpretable, and football-specific probability model designed to support real decision-making—not just accuracy numbers.

---

## **🎯 Goal of the Project**

**Build a reliable model for in-game decision support.**

Instead of “will the kicker make it?”, coaches need:

> *“What is the probability we make this kick—right here, right now?”*

This enables smarter choices between attempting the field goal, going for it, punting, or managing the clock differently. The modeling pipeline prioritizes **probabilistic calibration**, **risk awareness**, and **real-world interpretability**.

---

## **🔧 Key Features & Engineering Work**

### **1. Feature Engineering**

Includes game context, weather, stadium, and kicker-history features such as:

* Kick distance
* Temperature & wind speed
* Roof type (dome vs outdoor)
* Playing surface (turf vs grass)
* Altitude & stadium effects
* Vegas win probability
* Rain & snow indicators
* Score differential
* 4th quarter / “buzzer-beater” pressure
* Kicker’s career FG% and attempt history

These capture *real, physics-driven* and *psychological* components of kicking.

---

### **2. Modeling Pipeline**

We trained and tuned multiple model families:

#### **Linear Models**

* Logistic Regression (L2)
* Bayesian Logistic Regression (strong shrinkage)
* Generalized Additive Model (GAM)
  **Strength:** Interpretability & stability

#### **Tree-Based Models**

* Bagging (bootstrap aggregated trees)
* LightGBM
* BART (Bayesian Additive Regression Trees)
  **Strength:** Nonlinear interactions (distance × wind × surface)

#### **Ensembles**

* Equal-weight model combinations
* **Weighted ensemble** using an exhaustive grid search

  * Best model: **0.10·Bagging + 0.35·LGBM + 0.40·GAM + 0.15·LR**

---

### **3. Calibration**

Raw model probabilities aren’t trustworthy. To fix this, we used **per-distance isotonic calibration**:

* 0–29 yards
* 30–39 yards
* 40–49 yards
* 50+ yards

Models learn different reliability in each bucket, producing smooth, realistic curves.

---

## **📊 Evaluation Metrics**

### **Primary Metric: Brier Score**

Measures the accuracy of probability predictions.

* Best single model (BART): **0.10993**
* Best equal-weight ensemble: **0.10981**
* **Best weighted ensemble: 0.10976**

### **Additional Metrics**

* **AUC** ∼0.75 across models
* **PR-AUC (miss)** ∼0.31
* **ECE@10 (calibration error)** as low as **0.011**

---

## **🎬 Live Experiment – Final Class Presentation**

To test the model “in the wild,” we selected **three extreme real NFL kicks**:

* 68-yard attempt indoors
* 43-yard windy playoff game-winner
* 52-yard cold-weather playoff attempt

These were deliberately chosen *outside the distribution* of typical NFL kicks.

> **Result:** The model was correct on only 1/3 → exactly the point. It performed well on normal kicks but struggled on extreme, low-data cases.

### **Takeaway from the Demo**

Models give **probability**, not truth. They support decision-making—but humans must interpret edge cases.

---

## **🧪 Demo Script**

Run predictions for any kick scenario:

```
python kicker_demo_model.py
```

Outputs calibrated probabilities for:

* Bagging
* LightGBM
* GAM
* Logistic Regression
* **Weighted Ensemble (final model)**

---

## **📚 Technologies Used**

* Python (NumPy, Pandas)
* scikit-learn
* LightGBM
* pyGAM
* IsotonicRegression
* Matplotlib / Seaborn
* Jupyter / Python scripts

---

## **👤 Author**

**Quinn Peters**
Duke University
Risk, Data, and Financial Engineering
Machine Learning · Sports Analytics · Decision Science

---

## **⭐ Final Note**

This project blends **machine learning**, **calibration**, **sports analytics**, and **decision theory** into a production-style pipeline. It demonstrates:

* Technical modeling depth
* Careful validation
* Strong communication of results
* The ability to design ML systems with **real-world use cases**

If you're a recruiter or engineer reviewing this repository, feel free to reach out!
