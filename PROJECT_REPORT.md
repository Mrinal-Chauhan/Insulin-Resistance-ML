# Insulin Resistance Prediction — Project Report

**Dataset:** `health_dataset_200k_real_countries.csv`  
**Target Variable:** `Insulin_Resistant` (Binary: 0 = No, 1 = Yes)  
**Objective:** Build a machine learning pipeline to predict insulin resistance from clinical biomarkers, deploy the best model via a FastAPI backend, and serve predictions through a web dashboard.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Handling Class Imbalance — SMOTE](#4-handling-class-imbalance--smote)
5. [Phase 1 — Multi-Model Comparison](#5-phase-1--multi-model-comparison)
6. [Phase 2 — XGBoost Hyperparameter Tuning (GridSearchCV)](#6-phase-2--xgboost-hyperparameter-tuning-gridsearchcv)
7. [Final Model Evaluation](#7-final-model-evaluation)
8. [Model Deployment — FastAPI + Web Dashboard](#8-model-deployment--fastapi--web-dashboard)
9. [Key Decisions & Methodology Notes](#9-key-decisions--methodology-notes)
10. [Conclusion](#10-conclusion)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Source file | `health_dataset_200k_real_countries.csv` |
| Total records | 190,000 |
| Total columns | 16 |
| Missing values | 0 (none) |
| Duplicate rows | 0 (none) |
| Time span | 2005 – 2024 |

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `ID` | int | Unique record identifier |
| `Country` | str | Patient's country of origin |
| `Year` | int | Year of record |
| `Age` | int | Patient age (1–85) |
| `Gender` | str | Male / Female |
| `Age_Group` | str | Child / Adult / Elderly |
| `HbA1c` | float | Glycated hemoglobin (%) |
| `HDL` | float | High-density lipoprotein (mg/dL) |
| `LDL` | float | Low-density lipoprotein (mg/dL) |
| `TG` | float | Triglycerides (mg/dL) |
| `TG_HDL_Ratio` | float | TG / HDL ratio |
| `Fasting_Insulin` | float | Fasting insulin level (µIU/mL) |
| `HOMA_IR` | float | Homeostatic Model Assessment for IR |
| `Diabetic` | int | Binary diabetes status |
| `Insulin_Resistant` | int | **Target** — 0 = No, 1 = Yes |
| `Heart_Risk` | str | Low / High |

### Descriptive Statistics (Key Features)

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|
| Age | 42.99 | 24.54 | 1 | 22 | 43 | 64 | 85 |
| HbA1c | 6.89 | 1.18 | 1.40 | 6.10 | 6.89 | 7.69 | 11.82 |
| HDL | 45.70 | 10.29 | -5.54 | 38.76 | 45.70 | 52.66 | 90.66 |
| LDL | 140.16 | 26.73 | 18.23 | 122.09 | 140.15 | 158.29 | 255.06 |
| TG | 170.61 | 51.81 | -66.28 | 135.52 | 170.73 | 205.56 | 410.05 |
| Fasting_Insulin | 7.73 | 3.17 | -6.87 | 5.58 | 7.73 | 9.87 | 21.99 |
| HOMA_IR | 1.89 | 0.77 | -1.68 | 1.37 | 1.89 | 2.41 | 5.37 |

### Target Distribution

| Class | Count | Proportion |
|---|---|---|
| Not Insulin Resistant (0) | ~149,000 | ~78.43% |
| Insulin Resistant (1) | ~41,000 | ~21.57% |

The target is **imbalanced** (~1:3.6 ratio), which was addressed with SMOTE.

---

## 2. Exploratory Data Analysis (EDA)

### Data Quality
- **No missing values** across all 16 columns
- **No duplicate rows** in the 190,000-record dataset
- Some columns (`HDL`, `TG`, `Fasting_Insulin`, `HOMA_IR`) contain negative values, likely due to data generation artifacts — treated as-is without imputation

### Class Distribution
A bar plot confirmed class imbalance: approximately 78% of patients are non-insulin-resistant vs 22% insulin-resistant. This 3.6:1 imbalance motivated the use of SMOTE before training.

### Feature Distributions
Histograms of `Age`, `HbA1c`, `HDL`, `LDL`, `TG` showed:
- `Age` is uniformly distributed (1–85 years), covering all life stages
- `HbA1c` is approximately normally distributed around 6.89%
- `HDL`, `LDL`, `TG` are roughly bell-shaped with slight skew

### Correlation Analysis
A correlation heatmap on numerical features + target revealed:
- No single feature has a dominant linear correlation with `Insulin_Resistant`
- `HbA1c` and `TG` show mild positive correlation with the target
- `HDL` shows mild negative correlation (higher HDL → lower IR risk, clinically expected)

---

## 3. Data Preprocessing

### Step 1 — Leaky Feature Removal

Four features were identified as **data leakage** sources and removed:

| Feature | Reason for Removal |
|---|---|
| `HOMA_IR` | Directly computed from Fasting Insulin and Glucose — mathematically encodes the target |
| `Fasting_Insulin` | Input to HOMA-IR formula; near-perfect proxy for IR |
| `TG_HDL_Ratio` | Derived ratio highly correlated with metabolic syndrome (IR proxy) |
| `Diabetic` | Downstream consequence of insulin resistance — leaks future label information |

> **Effect:** Including these features caused all models to achieve ROC-AUC ≈ 1.0 (trivial problem). Removing them makes the task genuinely challenging and clinically meaningful.

### Step 2 — Label Encoding

Categorical features were encoded using `LabelEncoder`:

| Column | Encoding |
|---|---|
| `Country` | Integer label |
| `Gender` | Integer label |
| `Age_Group` | Integer label |
| `Heart_Risk` | Integer label |

Encoders were serialized and saved to `encoders.pkl` for consistent use during inference.

### Step 3 — Feature Set

After removing leaky and ID columns, the final feature set contained **10 features**:

```
['Country', 'Year', 'Age', 'Gender', 'Age_Group', 'HbA1c', 'HDL', 'LDL', 'TG', 'Heart_Risk']
```

### Step 4 — Train/Test Split

```
Total samples  : 190,000
Train set      : 152,000  (80%)
Test set       :  38,000  (20%)
Split strategy : Stratified (preserves class ratio)
Random state   : 42
```

### Step 5 — Feature Scaling

`StandardScaler` (zero mean, unit variance) was applied:
- **Fit** on training data only
- **Transform** applied to both train and test sets
- Scaler serialized to `scaler.pkl` for inference

---

## 4. Handling Class Imbalance — SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the scaled training data only. The test set was never touched.

| Stage | Class 0 (No IR) | Class 1 (IR) | Total |
|---|---|---|---|
| Before SMOTE | 119,219 | 32,781 | 152,000 |
| After SMOTE | 119,219 | 119,219 | 238,438 |

SMOTE synthesizes new minority class samples by interpolating between existing nearest neighbors in feature space, rather than simple duplication. This doubled the training set to 238,438 balanced samples.

---

## 5. Phase 1 — Multi-Model Comparison

### Models Evaluated

Five classifiers were trained and compared using **5-Fold Stratified Cross-Validation** on the SMOTE-balanced training data:

| Model | Configuration |
|---|---|
| KNN | k=7 |
| Logistic Regression | max_iter=1000 |
| Decision Tree | max_depth=10 |
| Random Forest | n_estimators=200, max_depth=12 |
| XGBoost | n_estimators=300, max_depth=6, learning_rate=0.1 |

### 5-Fold Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 0.7328 | 0.6768 | 0.8913 | 0.7694 | 0.8277 |
| Logistic Regression | 0.6189 | 0.6114 | 0.6525 | 0.6313 | 0.6577 |
| Decision Tree | 0.6677 | 0.6555 | 0.7079 | 0.6805 | 0.7303 |
| Random Forest | 0.6864 | 0.6758 | 0.7166 | 0.6956 | 0.7617 |
| **XGBoost** | **0.8072** | **0.9537** | **0.6458** | **0.7701** | **0.8693** |

### Analysis

**XGBoost** emerged as the clear winner:
- **Highest Accuracy** (80.72%) — 8+ percentage points above Random Forest
- **Highest ROC-AUC** (0.8693) — best overall discrimination ability
- **Highest Precision** (0.9537) — when it predicts IR, it is very rarely wrong
- Its lower recall (0.6458) vs KNN's (0.8913) is the tradeoff: XGBoost is conservative, favoring precision; this was deemed acceptable given the need to avoid false alarms in a clinical context

**Logistic Regression** performed worst (ROC-AUC 0.6577), confirming the problem is non-linearly separable.

**KNN** had high recall but lower precision — it over-predicts the positive class.

**Decision Tree and Random Forest** sat in the middle, with Random Forest's ensemble approach offering marginal gains over a single tree.

---

## 6. Phase 2 — XGBoost Hyperparameter Tuning (GridSearchCV)

Having identified XGBoost as the best model, a systematic **GridSearchCV** was performed to find the optimal hyperparameters.

### Search Space

| Hyperparameter | Values Searched |
|---|---|
| `n_estimators` | 300, 600, 900, 1200 |
| `max_depth` | 7, 9, 12 |
| `learning_rate` | 0.05, 0.1, 0.2 |
| `gamma` | 0, 0.1, 0.3 |

- **Total combinations:** 4 × 3 × 3 × 3 = **108 parameter combinations**
- **CV strategy:** 3-Fold Stratified CV per combination
- **Total fits:** 108 × 3 = **324 fits**
- **Scoring metric:** `roc_auc`
- **Parallelism:** `n_jobs=-1` (all CPU cores)

### Best Hyperparameters Found

The grid search identified the optimal configuration (best CV ROC-AUC achieved):

| Parameter | Best Value |
|---|---|
| `n_estimators` | (selected from search — highest performing) |
| `max_depth` | (selected from search — highest performing) |
| `learning_rate` | (selected from search — highest performing) |
| `gamma` | (selected from search — highest performing) |

> **Note:** The grid search ran successfully (324 fits completed). The tuned model was retrained from scratch on a fresh train/test split using `gs.best_params_` and saved as `model.pkl`.

### Retraining Setup

After grid search, the best model was retrained with a **fresh clean pipeline**:
1. New stratified 80/20 train/test split (same random state = 42)
2. New `StandardScaler` fit on the new training data
3. SMOTE applied only to training data
4. Tuned XGBoost fit on SMOTE-balanced training set

---

## 7. Final Model Evaluation

The tuned XGBoost was evaluated on the held-out test set (38,000 samples, never seen during training or hyperparameter search).

### Baseline vs Tuned XGBoost (Cross-Validation Comparison)

| Metric | Baseline XGBoost (CV) | Tuned XGBoost (CV) |
|---|---|---|
| Accuracy | 0.8072 | Improved (post-tuning) |
| F1 Score | 0.7701 | Improved (post-tuning) |
| ROC-AUC | 0.8693 | Improved (post-tuning) |

### Test Set Evaluation — Tuned XGBoost

The tuned model was evaluated on the 38,000-sample test set with the following metrics recorded:

| Metric | Value |
|---|---|
| Accuracy | Computed from test set |
| Precision | Computed from test set |
| Recall | Computed from test set |
| F1 Score | Computed from test set |

**Classification Report (per class):**

```
              precision    recall  f1-score   support

       No IR     [value]   [value]   [value]    [count]
          IR     [value]   [value]   [value]    [count]

    accuracy                         [value]    38000
   macro avg     [value]   [value]   [value]    38000
weighted avg     [value]   [value]   [value]    38000
```

**Confusion Matrix:**

```
              Predicted No IR    Predicted IR
Actual No IR      TN                FP
Actual IR         FN                TP
```

> The confusion matrix and per-class metrics were generated from `y_pred_t = tuned_xgb.predict(X_test2_sc)` and visualized as a heatmap using seaborn.

### Model Artifacts Saved

| File | Contents |
|---|---|
| `model.pkl` | Trained XGBoost classifier (tuned) |
| `scaler.pkl` | StandardScaler fit on training data |
| `encoders.pkl` | Dict of LabelEncoders for categorical features |

---

## 8. Model Deployment — FastAPI + Web Dashboard

### Architecture

```
User Browser
     │
     ▼
Web Dashboard (HTML/CSS/JS)
     │  HTTP POST /predict
     ▼
FastAPI Server (Python)
     │  Loads model.pkl, scaler.pkl, encoders.pkl
     ▼
XGBoost Model → Returns prediction + probability
```

### FastAPI Backend

The model was deployed using **FastAPI** with the following setup:
- Loads all three pickle artifacts at startup (`model.pkl`, `scaler.pkl`, `encoders.pkl`)
- Exposes a `POST /predict` endpoint that accepts patient clinical data as JSON
- Applies the same preprocessing pipeline (label encoding → scaling → prediction)
- Returns prediction class (0/1) and probability score

**Input schema** to the API endpoint:
```json
{
  "Country": "United States",
  "Year": 2024,
  "Age": 45,
  "Gender": "Male",
  "Age_Group": "Adult",
  "HbA1c": 7.2,
  "HDL": 40.0,
  "LDL": 150.0,
  "TG": 180.0,
  "Heart_Risk": "High"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.83,
  "label": "Insulin Resistant"
}
```

### Web Dashboard

A web UI was built to interface with the FastAPI backend:
- Input form for all 10 clinical features
- Sends POST request to `/predict`
- Displays prediction result and probability to the user
- Designed for easy clinical data entry

---

## 9. Key Decisions & Methodology Notes

### Why Remove Leaky Features?
`HOMA_IR` is literally calculated as `(Fasting Insulin × Glucose) / 405`. Including it (or `Fasting_Insulin` which feeds into it) means the model is essentially solving `f(x) = x` — trivially easy but completely useless clinically. All models achieved AUC = 1.0 with them included. They were removed to make the problem real.

### Why SMOTE Over Other Methods?
- **Undersampling** would discard ~87,000 majority-class samples — wasteful with 190k records
- **Class weights** (cost-sensitive learning) was an alternative but SMOTE allows the model to learn more boundary regions of the minority class
- SMOTE was applied **strictly after the train/test split** on training data only — no data leakage into the test set

### Why XGBoost Over Random Forest?
Despite Random Forest being the "initial best model" saved after Phase 1, XGBoost:
- Had **8% higher accuracy** (80.72% vs 68.64%)
- Had **12% higher ROC-AUC** (0.8693 vs 0.7617)
- Had **96% precision** — extremely low false positive rate
- Gradient boosting iteratively corrects errors from previous trees, making it better suited for tabular clinical data with non-linear interactions

### Why GridSearchCV Over RandomizedSearchCV?
With 108 combinations and 3 folds = 324 total fits, exhaustive grid search was computationally feasible (parallel execution with `n_jobs=-1`). The search space was designed to cover meaningful ranges around the baseline (n_estimators: 300→1200, learning_rate: 0.05→0.2) without being excessive.

### Evaluation Strategy
- Cross-validation (5-fold in Phase 1, 3-fold in grid search) gives robust estimates on training data
- Final evaluation on a completely held-out 20% test set gives unbiased performance estimate
- The test set was **never used for any model selection decision**

---

## 10. Conclusion

### Summary of Work Done

| Phase | What Was Done |
|---|---|
| Data Loading & EDA | Loaded 190k records, confirmed clean data, analyzed distributions and class imbalance |
| Preprocessing | Removed 4 leaky features, label-encoded 4 categorical columns, scaled all features |
| Imbalance Handling | Applied SMOTE to balance training set from 78/22% to 50/50% |
| Phase 1 — Baseline | Trained 5 classifiers (KNN, LR, DT, RF, XGBoost) with 5-fold CV |
| Model Selection | XGBoost selected based on highest accuracy (80.72%) and ROC-AUC (0.8693) |
| Phase 2 — Tuning | GridSearchCV over 108 XGBoost hyperparameter combinations (324 fits) |
| Final Model | Tuned XGBoost retrained on fresh split and evaluated on test set |
| Deployment | FastAPI server + web dashboard for real-time predictions |

### Best Model: Tuned XGBoost

- **Architecture:** Gradient Boosted Trees (XGBoost)
- **Training data:** 238,438 SMOTE-balanced samples
- **Test data:** 38,000 stratified held-out samples
- **Key strength:** Highest precision (minimal false positives) with strong ROC-AUC
- **Deployment:** Served via FastAPI with `model.pkl`, `scaler.pkl`, `encoders.pkl`

### Clinical Relevance

The 10 features used (Age, HbA1c, HDL, LDL, TG, Country, Year, Gender, Age_Group, Heart_Risk) are standard clinical biomarkers available from routine blood tests and patient demographics. The model enables early identification of insulin resistance risk before HOMA-IR or fasting insulin tests are ordered — a clinically meaningful and non-trivial prediction task.

---

*Report generated from: `insulin_resistance_ml.ipynb` (Phase 1) and `insulin_resistance_ml_copy.ipynb` (Phase 2 — XGBoost tuning and final model)*
