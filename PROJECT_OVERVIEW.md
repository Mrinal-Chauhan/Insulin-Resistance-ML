# Insulin Resistance Prediction — Project Overview

**Notebook:** `insulin_resistance_ml.ipynb`  
**Dataset:** `health_dataset_200k_real_countries.csv`  
**Task:** Binary classification — predict whether a patient is insulin resistant (`Insulin_Resistant`: 0 = No, 1 = Yes)

---

## 1. Dataset

| Property | Value |
|---|---|
| Rows | 190,000 |
| Columns | 16 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Time span | 2005–2024 |
| Countries | 190 |
| Target class split | 78.4% Not Resistant / 21.6% Resistant (~3.6:1 imbalance) |

### Features

| Column | Type | Description |
|---|---|---|
| `Country` | categorical | Patient's country (190 unique) |
| `Year` | int | Year of record |
| `Age` | int | Patient age (1–85) |
| `Gender` | categorical | Male / Female |
| `Age_Group` | categorical | Child / Teen / Adult / Elderly |
| `HbA1c` | float | Glycated haemoglobin (%) |
| `HDL` | float | HDL cholesterol (mg/dL) |
| `LDL` | float | LDL cholesterol (mg/dL) |
| `TG` | float | Triglycerides (mg/dL) |
| `TG_HDL_Ratio` | float | Triglyceride-to-HDL ratio |
| `Fasting_Insulin` | float | Fasting insulin level |
| `HOMA_IR` | float | Homeostatic Model Assessment of Insulin Resistance |
| `Diabetic` | binary | 0 / 1 |
| `Heart_Risk` | categorical | Low / Medium / High |
| `Insulin_Resistant` | binary | **Target label** |

---

## 2. Pipeline Summary

### Stage 1 — Exploratory Data Analysis (EDA)
- Schema inspection: zero nulls, zero duplicates across all 16 columns
- Target distribution: bar and pie charts showing 78.4 / 21.6 class split
- Categorical features vs target: grouped bar charts for `Gender`, `Age_Group`, `Heart_Risk`, `Diabetic`
- Numerical feature distributions: overlapping histograms by class for all 8 numeric features
- Box plots for outlier detection per feature per class
- Pearson correlation heatmap across all numeric and binary features
- Feature correlation with target: ranked horizontal bar chart
- Pair plot of key features (`HOMA_IR`, `Fasting_Insulin`, `TG_HDL_Ratio`, `HbA1c`)
- Year-wise IR prevalence trend and top-15 countries by IR rate

### Stage 2 — Preprocessing
- **Label Encoding** of 4 categorical columns (`Country`, `Gender`, `Age_Group`, `Heart_Risk`) — encoders saved as `.pkl` files
- **Stratified Train/Test Split:** 80% train (152,000 rows) / 20% test (38,000 rows), class balance preserved
- **Standard Scaling** (`StandardScaler`) fitted on train, applied to test — saved as `scaler.pkl`

### Stage 3 — PCA (Dimensionality Reduction)
- Full PCA fitted on scaled training data
- 14 features reduced to **11 principal components** retaining 98.04% variance (≥95% threshold)
- Comparison run: RF on original features (AUC = 1.0000) vs RF on PCA (AUC = 0.9975)
- Decision: final models trained on **original scaled features** (PCA incurs small accuracy cost)
- Saved as `pca.pkl` for reference

### Stage 4 — Model Training & Evaluation (All Features)

Five classifiers trained with **Stratified 5-Fold Cross-Validation**:

| Model | CV Accuracy | CV F1 | CV ROC-AUC |
|---|---|---|---|
| Random Forest | 0.9998 | 0.9996 | 1.0000 |
| Logistic Regression | 0.9992 | 0.9982 | 1.0000 |
| XGBoost | 0.9991 | 0.9980 | 1.0000 |
| Decision Tree | 0.9997 | 0.9993 | 0.9996 |
| K-Nearest Neighbors | 0.9601 | 0.9027 | 0.9913 |

Test set results (38,000 rows):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.9997 | 0.9993 | 1.0000 |
| Logistic Regression | 0.9993 | 0.9984 | 1.0000 |
| XGBoost | 0.9989 | 0.9976 | 1.0000 |
| Decision Tree | 0.9993 | 0.9984 | 0.9987 |
| K-Nearest Neighbors | 0.9630 | 0.9102 | 0.9924 |

### Stage 5 — GridSearchCV (Random Forest)
- Grid search on a stratified 30,000-row subsample
- Search space: `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {8, 12, 16, None}, `min_samples_split` ∈ {2, 5, 10}, `max_features` ∈ {sqrt, log2}
- **Best params:** `n_estimators=200, max_depth=12, max_features='sqrt', min_samples_split=5`
- Tuned model re-fit on full training set; AUC remained 1.0000 (confirming leakage, not model quality)

---

## 3. Data Leakage — Investigation & Fix

### Problem

After training all five models, every model achieved **ROC-AUC = 1.0000** on both cross-validation and the held-out test set. A perfect AUC across five diverse models — including a simple Logistic Regression — is a clear red flag. It means the feature set contains variables that effectively hand the model the answer, making learning trivial.

### Root Cause

The target label `Insulin_Resistant` was found to be derived directly from a single hard threshold on `HOMA_IR`:

```
Insulin_Resistant = 1  if  HOMA_IR >= 2.5
Insulin_Resistant = 0  if  HOMA_IR <  2.5
```

This was verified empirically:

| HOMA_IR threshold rule | Accuracy vs target |
|---|---|
| HOMA_IR >= 2.0 | 76.99% |
| **HOMA_IR >= 2.5** | **99.81%** |
| HOMA_IR >= 3.0 | 86.15% |

The `Insulin_Resistant=0` group has `HOMA_IR` max = 2.50 exactly; the `Insulin_Resistant=1` group starts at 2.50. The distributions do not overlap at all — a perfect step function.

### Leaky Features Identified

| Feature | Reason removed |
|---|---|
| `HOMA_IR` | **Primary source of leakage.** Target label derived from `HOMA_IR >= 2.5`. Pearson correlation with target: **0.712** |
| `Fasting_Insulin` | Direct component of the HOMA-IR formula: `HOMA_IR = (Fasting_Insulin × Fasting_Glucose) / 405`. Distributions for the two classes are completely separated. Correlation: **0.712** |
| `TG_HDL_Ratio` | A well-known clinical surrogate marker for insulin resistance; acts as a secondary proxy in this dataset |
| `Diabetic` | Type 2 diabetes and insulin resistance are near-synonymous conditions — including this creates circular reasoning |

### Fix Applied

The four leaky features were dropped and the entire pipeline was re-run from scratch on the **10 remaining clean features**: `Country`, `Year`, `Age`, `Gender`, `Age_Group`, `HbA1c`, `HDL`, `LDL`, `TG`, `Heart_Risk`.

### Results After Fix (Test Set — 38,000 rows)

| Model | Accuracy (leaky) | Accuracy (clean) | F1 (leaky) | F1 (clean) | ROC-AUC (leaky) | ROC-AUC (clean) |
|---|---|---|---|---|---|---|
| Random Forest | 0.9997 | 0.7843 | 0.9993 | 0.0005 | 1.0000 | 0.6565 |
| Logistic Regression | 0.9993 | 0.7843 | 0.9984 | 0.0000 | 1.0000 | 0.6588 |
| XGBoost | 0.9989 | 0.7836 | 0.9976 | 0.0224 | 1.0000 | 0.6504 |
| Decision Tree | 0.9993 | 0.7804 | 0.9984 | 0.0151 | 0.9987 | 0.6464 |
| K-Nearest Neighbors | 0.9630 | 0.7587 | 0.9102 | 0.1517 | 0.9924 | 0.5834 |

The ROC-AUC values correctly collapse from 1.0 to the 0.58–0.66 range. This reflects the true difficulty of the problem — the 10 remaining features (demographics + basic lipid panel) have very weak discriminative power for insulin resistance once the direct diagnostic markers are removed.

**Note on near-zero F1:** After removing the leaky features, most models default to predicting the majority class (Not Resistant, 78.4%) almost exclusively, producing near-zero F1 for the minority class. This is expected on a class-imbalanced dataset with weak remaining features — in a real project the next step would be to apply class balancing (SMOTE, class_weight='balanced') and threshold tuning.

---

## 4. Saved Artifacts

| File | Size | Contents |
|---|---|---|
| `model.pkl` | ~2,145 KB | Tuned `RandomForestClassifier` trained on all 14 features (leaky — for reference only) |
| `model_clean.pkl` | ~42,988 KB | `RandomForestClassifier` trained on 10 clean features (use this for honest inference) |
| `scaler.pkl` | ~1 KB | `StandardScaler` fitted on all 14 features |
| `scaler_clean.pkl` | ~1 KB | `StandardScaler` fitted on 10 clean features (use with `model_clean.pkl`) |
| `pca.pkl` | — | `PCA` transformer (11 components, 98% variance) |
| `label_encoder_Country.pkl` | ~2.3 KB | `LabelEncoder` for `Country` |
| `label_encoder_Gender.pkl` | ~0.3 KB | `LabelEncoder` for `Gender` |
| `label_encoder_Age_Group.pkl` | ~0.3 KB | `LabelEncoder` for `Age_Group` |
| `label_encoder_Heart_Risk.pkl` | ~0.3 KB | `LabelEncoder` for `Heart_Risk` |

> **Note:** `model.pkl` was saved in Section 5 using all 14 features (before the leakage fix). For production use, retrain using the clean feature set from Section 8 and save a new artifact.

---

## 5. Inference Demo

The notebook (Section 6) demonstrates end-to-end inference:

1. Load `model.pkl`, `scaler.pkl`, all `label_encoder_*.pkl` files
2. Encode categorical fields using saved label encoders
3. Scale using saved scaler
4. Call `model.predict()` and `model.predict_proba()`

**Example patient:** 45-year-old male from India, HbA1c=6.5, TG=200, HOMA_IR=3.5, Fasting_Insulin=15  
**Prediction:** Insulin Resistant — probability 98.8%  
*(Note: this uses the leaky model; prediction is driven entirely by HOMA_IR=3.5 ≥ 2.5)*

---

## 6. Key Findings

1. **The dataset is synthetically generated.** Perfect cleanliness (zero nulls, zero duplicates, exactly 1,000 rows per country) and a hard threshold label are strong indicators of a synthetic dataset built from a rule-based generator.

2. **HOMA_IR alone determines the label.** `Insulin_Resistant` is directly derived from `HOMA_IR >= 2.5`. Any model given `HOMA_IR` will achieve near-perfect performance trivially.

3. **Fasting_Insulin is equally leaky.** Its distributions for the two classes have zero overlap, making it a near-perfect proxy for the HOMA_IR threshold.

4. **After removing leaky features, the problem becomes genuinely hard.** Predicting insulin resistance from demographics and a standard lipid panel (HbA1c, HDL, LDL, TG) is a realistic, clinically relevant challenge. The best ROC-AUC achieved on clean features is **0.6588** (Logistic Regression), with near-zero F1 scores — most models collapse to predicting the majority class due to the 3.6:1 class imbalance and weak remaining signal. Class balancing would be the natural next step.

5. **PCA analysis:** 14 features compress to 11 principal components at 95% variance threshold, retaining 98.04% variance. The small dimensionality reduction has minimal practical benefit here but confirms that the features carry moderate collinearity.

---

## 7. Notebook Structure

| Section | Title |
|---|---|
| 1 | Load Data |
| 2 | EDA — Overview (2.1–2.8) |
| 3 | Preprocessing (3.1–3.4) |
| 3.5 | PCA — Dimensionality Reduction |
| 4 | Model Training & Evaluation — All Features (4.1–4.10) |
| 5 | Save Best Model |
| 6 | Inference Demo |
| 7 | Summary |
| **8** | **Data Leakage Investigation & Fix (8.1–8.6)** |
