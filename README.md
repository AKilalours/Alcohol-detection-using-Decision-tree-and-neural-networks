<div align="center">

# 🧠 AI-Powered Drug Consumption & Alcohol Use Prediction

### Ensemble ML System — Decision Tree · XGBoost · Neural Network · SHAP Explainability

**Course:** LIU — AI 680 | Final Project — Team 8

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green?style=flat-square)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.42.1-purple?style=flat-square)](https://shap.readthedocs.io)
[![Models](https://img.shields.io/badge/models-3%20ensemble-blueviolet?style=flat-square)]()

> ⚠️ **Academic project. Not for clinical or legal use. Research and education only.**

</div>

---

## 🎯 Goal & SLOs

Predict whether an individual is an **alcohol (or drug) consumer** from personality traits, demographics, and behavioural scores — using an **ensemble of three ML models** with majority-vote decision, SHAP explainability, and optimal threshold tuning.

| SLO | Target | Achieved |
|---|---|---|
| **Ensemble majority vote** | 3 models agree on final prediction | ✅ DT + XGBoost + NN |
| **Class imbalance handling** | Balanced training set | ✅ SMOTETomek / SMOTE fallback |
| **Optimal threshold** | Youden's J statistic per model | ✅ Per-model threshold stored |
| **Explainability** | SHAP attribution per prediction | ✅ Summary plots + per-model SHAP |
| **Monitoring artefacts** | Metrics saved per run | ✅ `monitoring/<model>_metrics.json` |

---

## 📐 Architecture & Data Flow

```
UCI Drug Consumption Dataset (URL load)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                  PREPROCESSING                           │
│  • Age bucketing → Age_Group (18–25, 26–35, 36–50, 50+)  │
│  • Binary encode drug columns (CL0 → 0, CL1+ → 1)        │
│  • LabelEncode: Gender, Education, Country, Ethnicity    │
│  • StandardScaler on all numerical features              │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                         │
│  Interaction features:                                   │
│    NexE = Nscore × Escore                                │
│    OexC = Oscore × Cscore                                │
│  Risk scores:                                            │
│    Risk_Score  = Impulsive × SS                          │
│    Risk_Score2 = Impulsive + SS                          │
│    Risk_Score3 = √(Impulsive² + SS²)                     │
│  Personality aggregates:                                 │
│    Personality_Score = mean(N, E, O, A, C)               │
│    Personality_Std   = std(N, E, O, A, C)                │
│    Personality_Range = max − min(N, E, O, A, C)          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│           FEATURE SELECTION (XGBoost-based)              │
│  SelectFromModel → threshold = 'median'                  │
│  scale_pos_weight = minority/majority ratio              │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│       CLASS BALANCING  (SMOTETomek → SMOTE fallback)     │
│  Oversample minority + undersample majority              │
│  Stratified 80/20 train-test split (random_state=42)     │
└──────────────────────┬───────────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         ▼             ▼              ▼
┌──────────────┐ ┌────────────┐ ┌──────────────────────┐
│ DECISION     │ │  XGBOOST   │ │   NEURAL NETWORK     │
│ TREE         │ │            │ │                      │
│              │ │ RandomizedS│ │  Dense(64) → ReLU    │
│ RandomizedSCV│ │ CV (50 iter│ │  Dropout(0.3)        │
│ (25 iter)    │ │ 5-fold CV) │ │  Dense(32) → ReLU    │
│ 5-fold CV    │ │ aucpr score│ │  Dropout(0.2)        │
│ balanced_acc │ │ avg_precis.│ │  Dense(1) → Sigmoid  │
│              │ │            │ │  Adam(lr=0.0005)     │
│ Pruning:     │ │ Params:    │ │  EarlyStopping(p=20) │
│ ccp_alpha    │ │ depth,lr,  │ │ class_weight=balanced│
│              │ │ subsample, │ │  300 epochs max      │
│              │ │ gamma,reg  │ │                      │
└──────┬───────┘ └─────┬──────┘ └──────────┬───────────┘
       └───────────────┼───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│          THRESHOLD OPTIMISATION (Youden's J)             │
│  Per model: find threshold maximising (TPR − FPR)        │
│  Threshold stored in monitoring/<model>_metrics.json     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              ENSEMBLE DECISION (Majority Vote)           │
│  votes = pred_dt + pred_xgb + pred_nn                    │
│  final = Consumer if votes ≥ 2 else Non-Consumer         │
│  Output: prediction + per-model confidence scores        │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│            EXPLAINABILITY & OUTPUTS                      │
│  • SHAP summary plots (XGBoost + NN)                     │
│  • Confusion matrices (all 3 models)                     │
│  • Precision-Recall curves (all 3 models)                │
│  • Decision Tree structure visualisation (max_depth=3)   │
│  • NN training history (PR-AUC, loss, precision/recall)  │
│  • Metrics JSON saved to monitoring/                     │
└──────────────────────────────────────────────────────────┘
```

**Trade-offs considered:**
- **Ensemble over single model:** majority vote across DT + XGBoost + NN reduces variance from any single model's threshold sensitivity on an imbalanced dataset.
- **Youden's J threshold vs 0.5:** fixed 0.5 threshold inflates precision but kills recall on imbalanced data; Youden's J finds the operating point that best balances TPR and FPR for each model independently.
- **SMOTETomek over SMOTE-only:** synthetic oversampling + Tomek link undersampling jointly cleans the decision boundary, reducing noise introduced by pure oversampling. SMOTE fallback ensures robustness if Tomek step fails.
- **RandomizedSearchCV over GridSearch:** 50-iteration random search over XGBoost's large hyperparameter space is more compute-efficient than exhaustive grid search while covering more diverse parameter combinations.
- **SHAP Explainer with Independent masker:** ensures SHAP values are computed against a realistic background distribution (100 samples), not a zero baseline, giving more reliable feature attributions.

---

## Why This Matters

Drug consumption prediction from personality and demographic data has public health applications — early identification of at-risk individuals enables targeted intervention. This system:

- Uses **no biometric or sensitive health records** — only personality scales (NEO-FFI derived), demographics, and impulsivity/sensation-seeking scores from the UCI dataset.
- Produces **interpretable predictions** via SHAP, so a counsellor or researcher can understand *why* a model flagged an individual.
- Is **uncertainty-aware**: outputs per-model confidence scores alongside the ensemble decision, not just a binary label.

---

## Dataset

**Source:** [UCI Drug Consumption Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data)  
Loaded directly from the UCI ML Repository — no local copy needed.

**Features used:**

| Category | Features |
|---|---|
| Demographics | Age, Gender, Education, Country, Ethnicity |
| Personality (NEO-FFI) | Nscore, Escore, Oscore, Ascore, Cscore |
| Behavioural | Impulsive (BIS-11), SS (ImpSS sensation seeking) |
| Engineered | NexE, OexC, Risk_Score, Risk_Score2, Risk_Score3, Personality_Std, Personality_Range, Personality_Score |

**Target:** `Alcohol` (binary) — `CL0` → Non-consumer (0), `CL1+` → Consumer (1).  
The same pipeline generalises to any of the 18 drug columns in the dataset.

---

## Models

### Decision Tree
- Hyperparameter search: `RandomizedSearchCV` (25 iterations, 5-fold stratified CV)
- Optimised for: `balanced_accuracy`
- Tuned params: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `ccp_alpha` (pruning)
- Saved to: `models/best_dt_model.pkl`

### XGBoost
- Hyperparameter search: `RandomizedSearchCV` (50 iterations, 5-fold stratified CV)
- Optimised for: `average_precision` (PR-AUC)
- Tuned params: `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, `scale_pos_weight`
- Saved to: `models/best_xgb_model.pkl`

### Neural Network
- Architecture: `Input → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)`
- Optimiser: Adam (lr=0.0005)
- Callbacks: EarlyStopping (patience=20, monitor=val_auc), ReduceLROnPlateau, ModelCheckpoint
- Class weighting: `compute_class_weight('balanced')` applied during training
- Saved to: `models/best_nn_model.keras`

---

## Evaluation Metrics

Beyond accuracy — because accuracy is misleading on imbalanced data:

| Metric | Why it matters here |
|---|---|
| **F2-score** | Weights recall 2× over precision — missing a consumer is costlier than a false alarm |
| **ROC-AUC** | Overall discrimination across all thresholds |
| **PR-AUC** | More informative than ROC-AUC under class imbalance |
| **Recall @ 90% Precision** | Practical operating point for screening tools |
| **Youden's J threshold** | Per-model optimal threshold rather than fixed 0.5 |

All metrics saved per run to `monitoring/<model>_metrics.json`.

---

## Output Artefacts

```
Plots/
├── confusion_matrix_dt.png        # DT confusion matrix
├── confusion_matrix_xgb.png       # XGBoost confusion matrix
├── confusion_matrix_nn.png        # NN confusion matrix
├── pr_curve_dt.png                # Precision-Recall curve — DT
├── pr_curve_xgb.png               # Precision-Recall curve — XGBoost
├── pr_curve_nn.png                # Precision-Recall curve — NN
├── shap_xgb.png                   # SHAP summary — XGBoost
├── shap_nn.png                    # SHAP summary — NN
├── decision_tree_structure.png    # DT visualisation (max_depth=3)
└── nn_training_history.png        # PR-AUC, loss, precision/recall curves

models/
├── best_dt_model.pkl
├── best_xgb_model.pkl
└── best_nn_model.keras

monitoring/
├── dt_metrics.json
├── xgb_metrics.json
└── nn_metrics.json
```

---

## 🔥 Postmortem: What Broke and How We Fixed It

### Issue 1 — SMOTETomek instability
**What happened:** SMOTETomek occasionally failed during resampling depending on environment and minority class size, causing a runtime crash.

**Root cause:** Tomek link removal is sensitive to very small minority class sizes; in some folds/runs the cleaning step produced errors downstream.

**Fix applied:** Wrapped in `try/except` with automatic SMOTE fallback. Both paths produce a balanced training set — SMOTETomek is preferred when available for cleaner decision boundaries.

---

### Issue 2 — XGBoost JSON serialisation crash in metrics saving
**What happened:** `json.dump(metrics)` failed because numpy float32/float64 values are not natively JSON serialisable.

**Root cause:** XGBoost and sklearn return numpy scalar types rather than Python native floats. `json.dump` does not handle these by default.

**Fix applied:** Explicit `.astype(float)` cast on `y_proba` and `float()` wrapping on all metric values before serialisation. All values in `metrics` dict are now Python-native types.

---

### Issue 3 — Neural network class weight API change
**What happened:** Passing `class_weight` as a numpy array caused a `ValueError` in newer Keras versions.

**Root cause:** Keras `model.fit()` expects `class_weight` as a `dict` mapping class index → weight, not an array.

**Fix applied:** Used `compute_class_weight` from sklearn and explicitly constructed `class_weight_dict = dict(zip(classes, weights))` before passing to `model.fit()`.

---

## 🛠️ MLOps & Infrastructure

### Reproducibility
- Global `SEED = 42` applied to numpy, TensorFlow, and all sklearn/XGBoost models.
- Stratified train-test split ensures consistent class ratios across runs.
- All model hyperparameter search results are deterministic via `random_state=SEED`.

### Experiment tracking
- Per-run metrics saved to `monitoring/<model>_metrics.json` — enables manual comparison across runs.
- Next step: integrate **MLflow** or **Weights & Biases** for automated metric logging, run comparison, and model registry.

### CI/CD awareness
- Planned: GitHub Actions workflow to run evaluation on every PR — block merge if any model's F2-score regresses > 5% from baseline.
- Eval gate: promote ensemble only if XGBoost PR-AUC ≥ 0.80 AND NN ROC-AUC ≥ 0.80.

### Deployment path
```
Train (local / Colab)
        │
        ▼
Serialise → .pkl (DT, XGBoost) + .keras (NN)
        │
        ▼
FastAPI endpoint
  └── POST /predict
        body: { age, gender, education, personality_scores, impulsive, ss }
        response: { prediction, confidence_dt, confidence_xgb, confidence_nn, ensemble }
        │
        ▼
Docker container → cloud deploy
        │
        ▼
Observability: log per-request confidence distribution
Alert if: avg ensemble confidence < 0.55 (data drift signal)
```

### Monitoring & alerts (planned)
- Log prediction confidence per model per request; alert if distribution shifts from training baseline.
- Scheduled weekly evaluation on held-out validation slice; alert if F2-score drops > 5%.
- Rollback: versioned `.pkl` / `.keras` files; rollback = swap model path in config.

---

## 🔁 Reliability: Caching, Fallbacks, Observability

| Concern | Approach |
|---|---|
| **Resampling failure** | SMOTETomek → automatic SMOTE fallback |
| **JSON serialisation** | All metrics cast to Python-native float before saving |
| **Model load failure** | Each model saved independently; pipeline continues if one fails to load |
| **Threshold sensitivity** | Per-model Youden's J threshold stored; not hardcoded to 0.5 |
| **Eval gates** | Promote only if: XGBoost PR-AUC ≥ 0.80 AND ensemble F2 ≥ 0.75 |
| **Rollback** | Versioned model checkpoints; rollback = config path swap |

---

## 🚀 Quickstart

```bash
git clone <your-repo-url>
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline (loads data from UCI, trains all 3 models, evaluates, saves outputs)
python main.py
```

Dataset loads automatically from the UCI ML Repository — no local download needed.

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow>=2.12
imbalanced-learn
shap
joblib
```

---

## 📁 Repository Structure

```
drug-consumption-prediction/
├── main.py                        # Full pipeline: load → engineer → train → evaluate → explain
├── requirements.txt
├── models/
│   ├── best_dt_model.pkl
│   ├── best_xgb_model.pkl
│   └── best_nn_model.keras
├── Plots/                         # All output visualisations
├── monitoring/                    # Per-run metrics JSON files
└── README.md
```

---

## 👥 Team & Contributions

This is a collaborative team project — all work split equally between both members.

| | **Akila Lourdes Miriyala Francis** | **Akilan Manivannan** |
|---|---|---|
| 📦 **Data & Preprocessing** | UCI dataset loading, binary drug encoding, age bucketing, LabelEncoder pipeline | StandardScaler integration, feature selection (XGBoost-based SelectFromModel) |
| 🔧 **Feature Engineering** | Interaction features (NexE, OexC), personality aggregates (Std, Range, Score) | Risk score variants (Risk_Score, Risk_Score2, Risk_Score3) |
| 🧠 **Modelling** | Decision Tree pipeline (RandomizedSearchCV, pruning via ccp_alpha) | XGBoost pipeline (50-iter RandomizedSearchCV, scale_pos_weight, aucpr) |
| 🏗️ **Neural Network** | Architecture design (Dense layers, Dropout, Sigmoid output), callback setup | Class weight computation fix, Adam optimiser tuning, EarlyStopping strategy |
| ⚖️ **Class Balancing** | SMOTETomek implementation, fallback logic (try/except → SMOTE) | Stratified train-test split, class distribution analysis |
| 📊 **Evaluation & Explainability** | Youden's J threshold optimisation, confusion matrices, PR curves | SHAP explainability (XGBoost + NN), decision tree structure visualisation |
| ⚡ **Ensemble & Inference** | Majority-vote ensemble logic, per-model confidence output | `predict_alcohol_use_ensemble()` function, full inference pipeline for new individuals |
| 📝 **Documentation & MLOps** | Postmortem write-up, monitoring JSON serialisation fix | Metrics saving pipeline, output directory structure, README |

> **Equal contribution** — all model selection, evaluation criteria, and design decisions were discussed and validated jointly.

---

<div align="center">
<sub>Built with scikit-learn · XGBoost · TensorFlow · SHAP · imbalanced-learn</sub><br/>
<sub>LIU AI 680 — Final Project — Team 8</sub>
</div>
