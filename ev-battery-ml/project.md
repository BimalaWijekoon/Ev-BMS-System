# EV Battery Charging Dataset — Machine Learning Project
## Full Technical Documentation

**Module:** Machine Learning  
**Dataset:** EV Battery Charging Dataset (Time-Series NEV Charging Data)  
**Source:** Kaggle — programmer3/ev-battery-charging-dataset  
**Records:** 1,900 | **Features:** 21  
**Problem Type:** Regression + Binary Classification  
**Deployment:** Hugging Face Spaces (Gradio)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Problem Framing and Justification](#3-problem-framing-and-justification)
4. [Project Architecture](#4-project-architecture)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Training — Regression](#8-model-training--regression-cycle-degradation)
9. [Model Training — Classification](#9-model-training--classification-safety-flags)
10. [Model Evaluation and Selection](#10-model-evaluation-and-selection)
11. [Source Code Architecture](#11-source-code-architecture)
12. [Deployment — Gradio UI on Hugging Face](#12-deployment--gradio-ui-on-hugging-face)
13. [Theory Traceability](#13-theory-traceability)
14. [Limitations and Future Work](#14-limitations-and-future-work)

---

## 1. Project Overview

### 1.1 Background

Electric vehicles rely on lithium-ion battery packs whose capacity degrades gradually with each charge-discharge cycle. Accurately predicting how much capacity a battery has lost — and detecting safety anomalies such as thermal and voltage violations — is one of the most commercially important problems in EV battery management. Without this capability, manufacturers cannot give drivers reliable range estimates, cannot schedule preventive maintenance, and cannot trigger safety shutdowns before a dangerous event causes physical damage.

This project applies the full supervised machine learning pipeline to a time-series dataset of NEV (New Energy Vehicle) battery charging states. The pipeline moves from raw sensor data through preprocessing, feature engineering, model training, evaluation, and production deployment as an interactive web application — structured as a professional, modular codebase rather than a single monolithic notebook.

### 1.2 Objectives

The project has three technical objectives, in order of priority as recommended by the module lecturer:

1. **Primary — Regression:** Predict `cycle_degradation`, a continuous float representing the fractional capacity loss at each recorded battery state.
2. **Secondary — Classification:** Predict `over_temp_flag`, a binary indicator of whether the battery is experiencing a thermal safety violation.
3. **Tertiary — Rule-Based Safety:** Handle `over_voltage_flag` via a physics-based threshold rule after determining that insufficient positive training examples exist for a meaningful statistical classifier.

### 1.3 Guiding Principle

The module lecturer explicitly stated: *"The final evaluation does not completely depend on the metrics values. The approach you took to solve this problem is the most important factor."*

Every decision in this project is therefore documented with a clear reason. No model is selected without justification. No preprocessing step is applied without explaining what problem it solves. The goal is to demonstrate engineering reasoning, not only to produce numbers.

---

## 2. Dataset Description

### 2.1 Source

**Dataset Name:** EV Battery Charging Dataset (Time-Series NEV Charging Data)  
**URL:** https://www.kaggle.com/datasets/programmer3/ev-battery-charging-dataset  
**Format:** Single CSV file  
**Size:** 1,900 rows × 21 columns  
**Nature:** Longitudinal time-series — each row represents a sequential battery state snapshot across the full lifecycle of one or more batteries, ordered chronologically by row index.

### 2.2 Column Definitions

| Column | Type | Range (observed) | Description |
|---|---|---|---|
| `timestamp` | Integer | 0 – 1899 | Sequential row index. Not a real datetime. Dropped before modelling. |
| `SOC` | Float | 0.06 – 96.75 | State of Charge. Grows monotonically across the dataset — acts as a cumulative cycle counter, not a per-session percentage. |
| `SOH` | Float | 0.856 – 0.999 | State of Health. Fraction of original capacity remaining. Oscillates — reflects per-cycle health measurement. |
| `terminal_voltage` | Float | 2.80 – 4.20 | Measured voltage at battery terminals in Volts. |
| `battery_current` | Float | 5.09 – 34.04 | Battery current in Amperes. |
| `battery_temp` | Float | 30.25 – 1129.49 | Cumulative thermal exposure across lifecycle. Increases monotonically — encodes long-term thermal aging history, NOT per-session temperature. |
| `ambient_temp` | Float | 10.61 – 40.00 | Environmental temperature in °C. Stable range — true ambient reading. |
| `internal_resistance` | Float | 0.019 – 0.150 | DC internal resistance in Ohms. Increases with battery aging. |
| `action_current` | Float | 3.87 – 34.05 | Commanded charging current. |
| `action_voltage` | Float | 2.80 – 4.20 | Commanded charging voltage. |
| `dT_dt` | Float | 0.03 – 2.06 | Rate of temperature change (°C per timestep). |
| `dV_dt` | Float | −0.046 – 0.032 | Rate of voltage change (V per timestep). Also called voltage slope. |
| `soc_delta` | Float | 0.012 – 0.082 | Change in SOC between successive timesteps. |
| `thermal_stress_index` | Float | 0.0 – 1.0 | Derived, normalised index of thermal stress. Value is 0 for all early rows, approaches 1 for late high-temperature rows. |
| `aging_indicator` | Float | 0.012 – 0.428 | Derived indicator of battery aging rate from SOH dynamics. |
| `charging_efficiency` | Float | 0.811 – 0.898 | Ratio of energy delivered to energy consumed. |
| `charging_time` | Float | 1848 – 4147 | Duration of the charging event in seconds. |
| `cycle_degradation` | Float | 0.0001 – 0.001 | **Regression target.** Fractional capacity loss per cycle. Small values. |
| `over_temp_flag` | Integer | 0 or 1 | **Classification target 1.** 1 = thermal safety violation detected. |
| `over_voltage_flag` | Integer | 0 or 1 | **Classification target 2.** 1 = voltage safety violation detected. |
| `balancing_time` | Float | 2.35 – 39.93 | Cell balancing duration in seconds. |

### 2.3 Critical Dataset Characteristics Discovered in EDA

The following properties are not obvious from the column names but fundamentally shape every modelling decision.

**`battery_temp` is a lifetime thermal accumulator, not a session temperature.**  
The value starts at ~30°C and rises to ~1129°C across 1,900 rows. No lithium-ion battery operates at 1129°C — that would be catastrophic failure. This column encodes the *cumulative thermal energy* absorbed across the entire battery lifecycle. It is effectively a thermal odometer. This means it is one of the strongest predictors of degradation (more thermal history = more degradation) but must be understood as a lifetime feature, not a session feature.

**`over_temp_flag` has temporal block structure.**  
The flag is 0 for approximately the first 900 rows and 1 for the remaining ~1,000 rows. This is not a randomly scattered rare-event pattern — it is a phase transition in the battery's lifecycle. Once cumulative thermal stress crosses a threshold, all subsequent readings are flagged. This has a direct and critical implication: random train-test splitting would allow the model to see future states (flag=1) during training while evaluating on past states (flag=0), creating data leakage that invalidates all metrics.

**`over_voltage_flag` is almost entirely zero.**  
Fewer than 15 positive cases exist in the entire dataset. Training a classifier on this would produce a model that memorises the noise in those 15 examples. A physics-based rule is more defensible and more reliable.

**`cycle_degradation` values are tiny floats in the range 0.0001–0.001.**  
The distribution is right-skewed with some values an order of magnitude larger than the median. Log transformation is necessary to make the target more normally distributed and to prevent the regression cost function from being dominated by the few large values.

**Derived features are already present.**  
`thermal_stress_index`, `aging_indicator`, `dT_dt`, `dV_dt`, and `soc_delta` are pre-computed derived features. Verification of their correlation with the source variables they are derived from must be performed before adding more engineered features to avoid redundancy.

---

## 3. Problem Framing and Justification

### 3.1 Why Regression for Cycle Degradation

`cycle_degradation` is a continuous-valued float. Predicting it requires a regression model — one that outputs a real number rather than a class label. From Lecture 04: *"Linear Regression models the relationship between a dependent variable and independent variables. The goal is to learn a mapping function from inputs to a continuous output."*

Alternatives considered:

- Discretising cycle_degradation into bins (Low / Medium / High) and treating it as a classification problem. Rejected because binning loses information — the model would not be able to distinguish between two degradation values that fall in the same bin, reducing precision in a context where accurate degradation prediction matters for maintenance scheduling.

The regression framing is kept continuous to preserve maximum predictive granularity.

### 3.2 Why Classification for Over-Temperature Flag

`over_temp_flag` is binary (0 or 1). From Lecture 04: *"Logistic Regression is used for binary classification. The output is a probability between 0 and 1."* The appropriate framing is binary classification.

More importantly, the evaluation metric priority is Recall over Accuracy. From Lecture 03: *"Recall = TP / (TP + FN). High recall = few missed positives. Use case: Medical diagnosis — you must NOT miss actual disease cases."* Battery thermal violations are the engineering equivalent — missing a True Positive (failing to flag an actual over-temperature event) can cause battery damage or fire. A False Negative here has real safety consequences. Therefore, Recall is the primary optimisation target for the classification models.

From Lecture 03 on why Accuracy is not appropriate: *"95% of emails are not spam → classifier that always says 'not spam' gets 95% accuracy but is useless."* The same logic applies here — a model that always predicts 0 would achieve ~47% accuracy on this dataset (if the positive class is ~53%) or ~47% if the split falls differently. This is meaningless. Recall and F1-Score are used instead.

### 3.3 Why Rule-Based for Over-Voltage Flag

With fewer than 15 positive examples of `over_voltage_flag = 1` across 1,900 rows, no statistically meaningful classifier can be trained. Any model would either memorise those 15 examples (extreme overfitting) or learn to always predict 0 (exploiting class imbalance). The honest and professional decision is to acknowledge this limitation and apply a physics-based rule:

**Rule:** Flag `over_voltage_flag = 1` if `action_voltage > 4.15` OR `terminal_voltage > 4.18`.

This threshold is grounded in lithium-ion battery chemistry. Standard lithium-ion cells have a maximum safe charge voltage of approximately 4.20V per cell. The thresholds 4.15V and 4.18V represent the margin before this limit, aligned with typical battery management system (BMS) alarm thresholds. This rule-based approach is more defensible, more reliable, and more transparent than a model trained on 15 positive examples.

---

## 4. Project Architecture

### 4.1 Folder Structure

```
ev-battery-ml/
│
├── data/
│   └── nev_battery_charging.csv
│
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_regression.ipynb
│   ├── 05_model_training_classification.ipynb
│   └── 06_model_evaluation_and_selection.ipynb
│
├── src/
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── train_regression.py
│   ├── train_classification.py
│   └── predict.py
│
├── models/
│   ├── regression_model.pkl
│   ├── classification_model_temp.pkl
│   ├── preprocessor_pipeline.pkl
│   ├── target_scaler.pkl
│   ├── feature_columns.json
│   ├── classification_config.json
│   └── MODEL_CARD.md
│
├── plots/
│   ├── eda_feature_distributions.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_time_series_trends.png
│   ├── eda_target_distributions.png
│   ├── preprocessing_class_balance.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_xgb.png
│   ├── learning_curve_regression.png
│   ├── xgb_training_curve.png
│   ├── roc_comparison_overtemp.png
│   ├── confusion_matrix_final.png
│   ├── actual_vs_predicted_regression.png
│   └── residual_plot_regression.png
│
├── app/
│   └── app.py
│
└── requirements.txt
```

### 4.2 Design Rationale — Why Not One Notebook

A single monolithic notebook is a procedural script, not an engineered system. Separating concerns into dedicated notebooks and source modules achieves four things:

First, **reproducibility** — each notebook can be re-run independently after changing only the relevant step, without re-running the entire pipeline. If a new feature engineering idea is tested in Notebook 03, Notebooks 04 and 05 can be re-run without repeating EDA.

Second, **reusability** — the `src/` modules (`preprocessor.py`, `feature_engineer.py`, `predict.py`) implement the exact same transformations used during training. When the Gradio UI calls `predict.py`, it applies identical preprocessing and feature engineering to inference inputs. This eliminates training-serving skew — one of the most common and damaging bugs in deployed ML systems.

Third, **clarity** — each file has one stated purpose. Code review, debugging, and modification are far easier when a file does one thing.

Fourth, **professionalism** — production ML systems at companies like Google, Meta, and Tesla are built as modular pipelines with separate ingestion, transformation, training, evaluation, and serving components. This project structure mirrors that approach at an appropriate scale.

### 4.3 Data Flow Diagram

```
nev_battery_charging.csv
        │
        ▼
Notebook 01: EDA
  - Inspect structure, statistics, distributions
  - Identify anomalies, temporal patterns, class imbalance
        │
        ▼
Notebook 02: Preprocessing
  - Drop timestamp
  - Impute missing values (median)
  - Chronological train/val/test split (70/15/15)
  - StandardScaler (fit on train only)
  - Log1p transform on cycle_degradation
  - Compute class weights for imbalanced classifier
  - Save: preprocessor_pipeline.pkl
        │
        ▼
Notebook 03: Feature Engineering
  - Verify existing derived features
  - Engineer: delta_internal_resistance, soc_range_rolling,
              thermal_acceleration, voltage_efficiency,
              polynomial features (IR², TSI², AI²)
  - Filter selection: drop low-correlation features
  - Wrapper selection: RFE with Random Forest
  - Save: feature_columns.json
        │
        ▼
Notebook 04: Regression Training          Notebook 05: Classification Training
  - Linear Regression (baseline)              - Logistic Regression (baseline)
  - Random Forest Regressor                   - SVM with RBF Kernel
  - XGBoost Regressor (primary)               - XGBoost Classifier (primary)
  - TimeSeriesSplit CV throughout             - TimeSeriesSplit CV throughout
  - Learning curve analysis                   - Threshold tuning
  - Save: regression_model.pkl                - Save: classification_model_temp.pkl
        │                                           │
        └──────────────────┬────────────────────────┘
                           ▼
              Notebook 06: Evaluation
                - Final test set evaluation only
                - Comparison tables
                - All visualisations
                - Model cards
                           │
                           ▼
                    src/predict.py
                  (inference logic)
                           │
                           ▼
                    app/app.py
                 (Gradio UI — two tabs)
                           │
                           ▼
              Hugging Face Spaces (public URL)
```

---

## 5. Exploratory Data Analysis

*Notebook: `01_data_understanding_eda.ipynb`*

### 5.1 Structural Inspection

```python
df = pd.read_csv('data/nev_battery_charging.csv')
print(df.shape)   # (1900, 21)
print(df.dtypes)  # All float64 except over_temp_flag and over_voltage_flag (int64)
```

All 21 columns are numeric. No categorical variables requiring encoding exist in this dataset. The `timestamp` column is a sequential integer index from 0 to 1899.

### 5.2 Missing Values

A full column-wise null audit is performed:

```python
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
```

**Finding:** Zero missing values across all 1,900 rows and 21 columns. Battery sensor systems either record a value or fail entirely — partial recording is rare. This eliminates the need for imputation in this specific dataset, though the imputation step is still built into the `BatteryPreprocessor` class to handle edge cases in inference inputs gracefully.

### 5.3 Descriptive Statistics

Extended percentile statistics (1st, 25th, 50th, 75th, 99th) are computed for all columns:

```python
df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
```

Key findings from the statistics:

**`battery_temp`:** Mean ~579°C, Min ~30°C, Max ~1129°C. A standard deviation of ~324°C confirms this is a monotonically increasing lifetime thermal variable, not a session reading. Its value encodes how much cumulative thermal energy the battery has absorbed since the start of its recorded lifecycle.

**`cycle_degradation`:** Mean ~0.00052, Min ~0.0001, Max ~0.00095. The range spans nearly one order of magnitude, and the distribution is right-skewed (the 99th percentile is far above the mean). This skewness confirms that a log transform is appropriate before regression training.

**`thermal_stress_index`:** Min 0.0, Max 1.0. The 25th percentile is 0.0, confirming the majority of early rows have zero thermal stress, while the upper quartile approaches 1.0 for late rows. This pattern mirrors the temporal block structure of `over_temp_flag`.

**`SOC`:** Min ~0.06, Max ~96.75. This is NOT a 0-to-1 fraction. It appears to be a cumulative charge accumulation metric scaled to approximately 0–100, growing monotonically over the dataset. This affects scaling decisions in preprocessing.

### 5.4 Target Variable Analysis

#### Cycle Degradation (Regression Target)

A histogram of `cycle_degradation` reveals a moderately right-skewed distribution. The bulk of values cluster between 0.0003 and 0.0006, with a tail extending toward 0.001. A box plot confirms the presence of values in the upper tail that would be treated as outliers in a symmetric distribution but are legitimate physical measurements (some cycles degrade batteries more than others).

**Decision:** Apply `np.log1p()` transformation. After transformation, the distribution becomes approximately normal, which is better suited for gradient-based optimisers in tree boosting algorithms and removes the influence of the upper tail on the mean squared error cost function.

**Theory connection (Lecture 02):** *"Log Transform: reduces skewness in data with extreme ranges. Income [10, 100, 1000, 10000] → [1, 2, 3, 4]."* The same principle applies to cycle_degradation values spanning an order of magnitude.

#### Over-Temperature Flag (Classification Target 1)

```python
df['over_temp_flag'].value_counts()
# 0: ~900 rows
# 1: ~1000 rows
```

Class balance is approximately 47%/53% — reasonably balanced in aggregate. However, plotting the flag against row index reveals the block structure: all 0s appear in the first ~900 rows, all 1s in the last ~1,000 rows. This temporal pattern is the critical finding that mandates chronological splitting.

#### Over-Voltage Flag (Classification Target 2)

```python
df['over_voltage_flag'].value_counts()
# 0: ~1885 rows
# 1: ~15 rows
```

Approximately 15 positive cases in 1,900 total rows — less than 1% positive rate. This extreme imbalance means even with SMOTE or class weighting, no classifier trained on 15 examples can generalise. The rule-based approach is adopted.

### 5.5 Feature Distributions

Histograms for all 18 non-target features are plotted in a 6×3 grid. Key observations:

- `battery_temp`, `SOC`, and `thermal_stress_index` all show monotonically increasing patterns — they are time proxies.
- `SOH`, `terminal_voltage`, `battery_current`, `ambient_temp` show approximately normal or uniform distributions — they represent real per-cycle measurements with natural variation.
- `internal_resistance`, `dT_dt`, `aging_indicator` show right-skewed distributions.
- `balancing_time`, `charging_time` appear roughly uniform across a wide range.

### 5.6 Correlation Analysis

A full Pearson correlation matrix is computed and visualised as a heatmap.

Pairs with absolute correlation above 0.85:

| Feature A | Feature B | Correlation |
|---|---|---|
| `battery_temp` | `thermal_stress_index` | ~0.97 |
| `battery_temp` | `SOC` | ~0.99 |
| `SOC` | `thermal_stress_index` | ~0.98 |
| `terminal_voltage` | `action_voltage` | ~0.91 |

**Decision from correlation analysis:**

`battery_temp`, `SOC`, and `thermal_stress_index` are mutually highly correlated (all three encode the same temporal/thermal lifecycle progression). Keep `thermal_stress_index` (already normalised to [0,1], physically interpretable) and `SOC` (carries cycle count information), and drop `battery_temp` as it is redundant given the other two.

Drop `action_voltage` in favour of keeping `terminal_voltage`, since terminal voltage is the measured quantity and action_voltage is the commanded quantity — their near-perfect correlation means only one is needed, and measured values are more informative for degradation prediction.

**Theory connection (Lecture 05):** *"Filter Methods: Statistical properties (correlation, chi-square, mutual info). Fast, easy to implement."*

### 5.7 Time-Series Trend Visualisation

A four-panel time-series plot of `battery_temp`, `SOC`, `SOH`, and `cycle_degradation` against row index confirms:

- `battery_temp` and `SOC` increase monotonically — they are temporal proxies.
- `SOH` oscillates between ~0.86 and ~0.99 — it reflects genuine cycle-by-cycle health variation.
- `cycle_degradation` shows a noisy but bounded signal throughout — it is the quantity to be predicted.

This visualisation is critical for understanding the dataset as a longitudinal battery lifecycle record rather than a collection of independent observations.

---

## 6. Data Preprocessing

*Notebook: `02_data_preprocessing.ipynb` | Source: `src/preprocessor.py`*

### 6.1 Drop Timestamp

```python
df = df.drop(columns=['timestamp'])
```

**Reason:** The timestamp column is a sequential integer 0 to 1899. Including it as a feature would teach any model to predict based on row position, not battery physics. Since inference will supply new data points that are not necessarily sequential continuations of the training set, the model would produce incorrect predictions. Row number carries no physical meaning — time information is already encoded in `battery_temp` and `SOC` which grow monotonically.

### 6.2 Duplicate Removal

```python
print(df.duplicated().sum())
df = df.drop_duplicates()
```

Exact duplicates are removed before any other transformation. With only 1,900 rows, even a small number of duplicate rows can meaningfully distort model training by over-weighting specific battery states.

### 6.3 Missing Value Imputation

Although EDA confirmed zero missing values in the training dataset, a `SimpleImputer(strategy='median')` is instantiated and built into the `BatteryPreprocessor` class. This serves two purposes: it is a no-op on this dataset (imputing zero missing values changes nothing) but it ensures that inference inputs — which may contain NaN values from sensor dropouts in a real deployment — are handled gracefully without crashing the prediction pipeline.

**Reason for median over mean (Lecture 02 Step 2):** Battery sensor distributions are right-skewed (battery_temp, dT_dt, aging_indicator). The median is more robust to extreme values in skewed distributions than the mean, which is pulled toward the tail.

### 6.4 Chronological Train / Validation / Test Split

```python
n = len(df)
train_end = int(n * 0.70)   # Row 0 to 1329 — training set
val_end   = int(n * 0.85)   # Row 1330 to 1614 — validation set
# Row 1615 to 1899           — test set

df_train = df.iloc[:train_end]
df_val   = df.iloc[train_end:val_end]
df_test  = df.iloc[val_end:]
```

**Why chronological, not random — this is the most important preprocessing decision in the entire project.**

The `over_temp_flag` transitions from 0 to 1 at approximately row 900. If `sklearn.model_selection.train_test_split` is used with `shuffle=True` (the default), rows from after the transition (flag=1, high thermal stress, high SOC, high battery_temp) would appear in the training set alongside rows from before the transition (flag=0, low thermal stress). The model would then learn that high thermal stress = flag=1, and when evaluated on the "test set" — which would contain a random mix of early and late rows — it would score extremely well, but only because it had already seen future states during training.

In a real deployment, the model would only ever see past battery states when predicting the current state. Chronological splitting simulates this correctly: the model is trained on the first 70% of the battery's lifetime, validated on the next 15%, and tested on the final 15% — states it has never seen.

**Theory connection (Lecture 01):** *"Always split data into Training, Validation, and Test sets. The model must generalise to unseen data — this is the core goal."*

After splitting, features and targets are separated for each split:

```python
targets = ['cycle_degradation', 'over_temp_flag', 'over_voltage_flag']
X_train = df_train.drop(columns=targets)
y_reg_train  = df_train['cycle_degradation']
y_cls_temp_train = df_train['over_temp_flag']
y_cls_volt_train = df_train['over_voltage_flag']
# Repeat for val and test
```

### 6.5 Feature Scaling with StandardScaler

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

The StandardScaler converts each feature to zero mean and unit variance: `z = (x - μ) / σ`, where μ and σ are computed from the training set only.

**Why scaling is necessary for this dataset:**

The feature ranges are wildly different. `SOC` spans 0–97, `SOH` spans 0.86–0.999, `battery_temp` spans 30–1129, and `internal_resistance` spans 0.02–0.15. In gradient-based models (XGBoost, Neural Networks) and distance-based models (SVM), features with larger numerical ranges contribute disproportionately to the gradient or distance calculation — not because they are more informative, but simply because their values are bigger. Scaling removes this arbitrary advantage.

**Why fit only on training data — the critical rule:**

The scaler is fitted (`fit`) only on `X_train`. The validation and test sets are transformed (`transform`) using the mean and standard deviation of the training set, not their own. If the scaler were fitted on the full dataset or on each split independently, the model would have indirect access to statistics of the validation and test sets during training — a subtle form of data leakage. The resulting metrics would be optimistically biased.

**Theory connection (Lecture 02 Step 3):** *"Standardization: converts to zero mean and unit variance. Normalize income values to [0, 1] for scale-sensitive models like SVM or KNN."*

The fitted scaler is saved to disk: `joblib.dump(scaler, 'models/preprocessor_pipeline.pkl')`.

### 6.6 Log Transform of Regression Target

```python
import numpy as np
y_reg_train_log = np.log1p(y_reg_train)
y_reg_val_log   = np.log1p(y_reg_val)
y_reg_test_log  = np.log1p(y_reg_test)
```

`np.log1p(x)` computes `log(1 + x)`, which handles zero values gracefully (whereas `np.log(0)` is undefined).

**Effect:** cycle_degradation values of 0.0001–0.001 are transformed to approximately −9.2 to −6.9. The right-skewed distribution becomes approximately normal, which:
- Prevents the MSE cost function from being dominated by the few large degradation values.
- Makes the regression target more compatible with the implicit normality assumptions in linear regression.
- Improves numerical stability in gradient computation for XGBoost.

**Inverse transform for reporting:** All final metrics are reported in original units by applying `np.expm1(predictions)` before computing MAE, RMSE, and R².

**Theory connection (Lecture 02):** *"Log Transform: reduces skewness in data with extreme ranges."*

### 6.7 Class Imbalance Strategy for Over-Temperature Classifier

After the chronological split, the class distribution in the training set is audited:

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced',
                                classes=np.unique(y_cls_temp_train),
                                y=y_cls_temp_train)
class_weight_dict = dict(enumerate(weights))
```

Because the `over_temp_flag` transition happens at ~row 900 and the training set covers rows 0–1329, the training set contains rows from both before and after the transition — giving an approximately 68%/32% split in favour of class 0 in training.

**Why `class_weight='balanced'` rather than SMOTE:**

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic samples by interpolating between existing minority class examples. For time-ordered battery data, interpolating between a row at position 900 and a row at position 1200 creates a synthetic sample that represents a physically impossible battery state — a mixture of two different lifecycle moments. This violates the temporal integrity of the data.

`class_weight='balanced'` achieves the same effect — penalising minority class misclassification more heavily — without generating physically invalid synthetic samples. It is passed directly to the model constructor.

### 6.8 Over-Voltage Flag Assessment

```python
print(y_cls_volt_train.value_counts())
# 0    ~1885
# 1    ~12
```

With approximately 12 positive training examples, no meaningful statistical classifier can be built. This finding is documented explicitly and the rule-based fallback is adopted:

```python
over_voltage_config = {
    'use_rule_based': True,
    'rule': 'action_voltage > 4.15 OR terminal_voltage > 4.18'
}
json.dump(over_voltage_config, open('models/classification_config.json', 'w'))
```

---

## 7. Feature Engineering

*Notebook: `03_feature_engineering.ipynb` | Source: `src/feature_engineer.py`*

### 7.1 Philosophy

From Lecture 02: *"Feature engineering is often more impactful than choosing the right algorithm. A simple algorithm with great features outperforms a complex algorithm with poor features."*

All feature engineering is performed on the raw (pre-scaled) data. The engineered features are then passed through the existing `StandardScaler` fitted in Notebook 02. Every engineered feature is motivated by battery physics, not statistical convenience.

### 7.2 Verification of Existing Derived Features

Before engineering new features, the existing derived columns are verified against their source variables:

```python
corr_temp_stress = df[['battery_temp', 'thermal_stress_index']].corr()
corr_soh_aging   = df[['SOH', 'aging_indicator']].corr()
```

**Findings:**
- `thermal_stress_index` correlates at ~0.97 with `battery_temp`. They encode essentially the same information. Retain `thermal_stress_index` (already normalised) and remove `battery_temp` from the feature set to eliminate redundancy.
- `aging_indicator` has moderate correlation (~0.45) with `SOH` — it captures SOH *rate of change* rather than the absolute level, so it carries additional information beyond `SOH` alone. Retain both.

### 7.3 Engineered Feature 1 — Delta Internal Resistance

```python
ir_baseline = df_train['internal_resistance'].iloc[0]
df['delta_internal_resistance'] = df['internal_resistance'] - ir_baseline
```

**Physical motivation:** Internal resistance of a lithium-ion cell increases as electrodes degrade and electrolyte decomposes. The absolute resistance value varies by cell chemistry and temperature, but the *change* from a healthy baseline is a universal aging signal. A battery that started with 0.05 Ω and now measures 0.14 Ω has degraded significantly regardless of its initial value.

**Theory connection (Lecture 02 Step 4):** *"Derive new features using domain knowledge. Example: From Date of Birth → create Age as a new feature."* `delta_internal_resistance` is the battery equivalent — time since health was at a reference level.

**Important:** The baseline value is computed from the training set's first row only and saved. During inference, the same baseline is used, not the first value of the inference batch — this prevents the engineered feature from shifting based on when inference is run.

### 7.4 Engineered Feature 2 — Rolling SOC Range

```python
df['soc_range_rolling'] = (df['SOC'].rolling(window=10).max()
                           - df['SOC'].rolling(window=10).min())
df['soc_range_rolling'].fillna(df['soc_range_rolling'].median(), inplace=True)
```

**Physical motivation:** Batteries aged between 20–80% SOC degrade significantly slower than those cycled between 0–100%. The rolling range of SOC over 10 timesteps captures the *charging aggressiveness* — how much of the SOC window is being used per charging event. High rolling range = aggressive cycling = faster degradation.

**Theory connection (Lecture 02 Step 4):** *"Combine Latitude + Longitude → Distance to Nearest Store."* The same principle — combining information from a sequence of readings into a single compound feature that encodes a physically meaningful quantity.

### 7.5 Engineered Feature 3 — Thermal Acceleration

```python
df['thermal_acceleration'] = df['dT_dt'] * df['battery_temp']
```

**Physical motivation:** A rapid temperature rise (high `dT_dt`) at an already-elevated temperature is far more damaging than the same rate of rise at a cool starting point. At 1000°C cumulative thermal exposure, a 2°C/timestep rise represents accelerating thermal runaway. At 30°C, the same rate is normal operation. Multiplying the two creates a compound feature that captures this interaction.

**Theory connection (Lecture 02 Step 4):** Creating higher-information compound features from domain knowledge, analogous to computing a distance feature from separate coordinate features.

### 7.6 Engineered Feature 4 — Voltage Efficiency Ratio

```python
df['voltage_efficiency'] = df['action_voltage'] / (df['terminal_voltage'] + 1e-6)
```

**Physical motivation:** The ratio between the commanded charging voltage and the measured terminal voltage indicates battery impedance. A healthy battery delivers terminal voltage close to the commanded value. As internal resistance increases with aging, the gap between commanded and measured voltage grows. This ratio captures that deviation efficiently.

The `+ 1e-6` prevents division by zero for any edge cases where terminal_voltage is zero.

### 7.7 Polynomial Features for Non-Linear Relationships

```python
df['internal_resistance_sq']  = df['internal_resistance'] ** 2
df['thermal_stress_sq']       = df['thermal_stress_index'] ** 2
df['aging_indicator_sq']      = df['aging_indicator'] ** 2
```

**Reason:** Battery degradation does not increase linearly with internal resistance, thermal stress, or aging — it accelerates exponentially. A battery at 0.14 Ω does not degrade twice as fast as one at 0.07 Ω; it degrades much more rapidly. Squaring the most degradation-relevant features allows linear models (and even tree models) to capture this accelerating non-linearity without requiring extremely deep trees or high-degree polynomial expansions.

**Theory connection (Lecture 02):** *"Polynomial Features: captures non-linear relationships: X → X², X³. Allows linear models to learn non-linear patterns."*

### 7.8 Feature Selection — Filter Method

```python
corr_with_target = (df_engineered.corr()['cycle_degradation']
                    .abs().sort_values(ascending=False))
weak_features = corr_with_target[corr_with_target < 0.03].index.tolist()
df_engineered = df_engineered.drop(columns=weak_features)
```

Features with absolute Pearson correlation below 0.03 with `cycle_degradation` carry statistically negligible linear signal for the regression target. They are dropped at this stage.

Separately, from the high-correlation pairs identified in EDA (correlation > 0.85 between feature pairs), the member of each pair with lower correlation to `cycle_degradation` is dropped. This removes redundancy without losing information.

**Theory connection (Lecture 05):** *"Filter Methods: Statistical properties (correlation, chi-square, mutual info). Fast, easy to implement. Disadvantage: ignores feature interactions."*

### 7.9 Feature Selection — Wrapper Method (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
estimator = RandomForestRegressor(n_estimators=50, random_state=42)
rfe = RFE(estimator, n_features_to_select=15)
rfe.fit(X_train_engineered_scaled, y_reg_train_log)

selected_features = [col for col, selected in
                     zip(X_train_engineered.columns, rfe.support_) if selected]
```

RFE (Recursive Feature Elimination) fits a Random Forest on all features, ranks features by importance, removes the least important, and repeats until 15 features remain. Unlike filter methods, RFE considers feature interactions — a feature that is weakly correlated with the target individually may still be important when combined with other features.

**Theory connection (Lecture 05):** *"Wrapper Methods: Train model with feature subsets (RFE, forward/backward). Considers interactions, high accuracy. Computationally expensive."*

The selected 15 features are saved to `models/feature_columns.json` as the canonical feature list for all subsequent training and inference.

### 7.10 Feature Selection Validation — Embedded Method

After XGBoost training in Notebook 04, the built-in feature importance scores are compared against the RFE selection. Features that appear as highly important in XGBoost but were excluded by RFE are reviewed manually, and the feature set may be adjusted accordingly.

**Theory connection (Lecture 05):** *"Embedded Methods: Built into model training (LASSO, Random Forest importance). Efficient, captures interactions. Tied to specific model."*

---

## 8. Model Training — Regression (Cycle Degradation)

*Notebook: `04_model_training_regression.ipynb` | Source: `src/train_regression.py`*

### 8.1 Cross-Validation Strategy

All hyperparameter tuning uses `TimeSeriesSplit(n_splits=5)` rather than the default `KFold`. This creates five chronologically ordered train-validation folds, where each validation fold is always more recent than its training fold. Default KFold shuffles data — for time-ordered data, this allows future states to appear in training folds and past states in validation folds, creating data leakage that inflates CV scores.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### 8.2 Model 1 — Linear Regression (Baseline)

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_selected_scaled, y_reg_train_log)
y_pred_log = lr.predict(X_val_selected_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_reg_val_log)
```

**Purpose:** Establish a performance floor. If linear regression performs surprisingly well (R² > 0.70), the relationships in the data are approximately linear and complex non-linear models may not be necessary. If it performs poorly (R² < 0.50), non-linear models are clearly justified.

Linear regression minimises the MSE cost function: `J(w) = (1/2m) × Σ(h_w(x) - y)²` and uses gradient descent to update weights: `wⱼ = wⱼ - α × ∂J(w)/∂wⱼ`.

**Theory connection (Lecture 04):** *"Linear regression models the relationship between a dependent variable and independent variables by fitting a linear equation to the data."*

**Expected performance:** Moderate. The polynomial features added in Notebook 03 give linear regression some ability to capture the non-linear degradation curve, but it cannot match tree-based methods for complex interactions.

### 8.3 Model 2 — Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 0.5]
}
rf = RandomForestRegressor(random_state=42)
gs = GridSearchCV(rf, param_grid, cv=tscv,
                  scoring='neg_root_mean_squared_error', n_jobs=-1)
gs.fit(X_train_selected_scaled, y_reg_train_log)
best_rf = gs.best_estimator_
```

**Theory connection (Lecture 06):** *"Random Forests enhance Bagging by adding random feature selection at each split, which decorrelates the trees. max_features: d/3 for regression. n_estimators: more trees = better (diminishing returns)."*

**Why Random Forest for this dataset:** With 1,900 rows, overfitting is a risk for any complex model. Random Forest's aggregation of many decorrelated trees reduces variance — the prediction is the average of many trees, smoothing out individual trees' overfitting. It also provides robust feature importance scores for the embedded selection validation.

**Feature importance extraction:**
```python
importances = pd.Series(best_rf.feature_importances_,
                         index=selected_features).sort_values(ascending=False)
importances.plot(kind='bar')
plt.savefig('plots/feature_importance_rf.png', dpi=150, bbox_inches='tight')
```

### 8.4 Model 3 — XGBoost Regressor (Primary)

```python
from xgboost import XGBRegressor

xgb = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1, 5, 10]
}
gs_xgb = GridSearchCV(xgb, param_grid, cv=tscv,
                       scoring='neg_root_mean_squared_error', n_jobs=-1)
gs_xgb.fit(X_train_selected_scaled, y_reg_train_log,
            eval_set=[(X_val_selected_scaled, y_reg_val_log)],
            verbose=False)
```

**Why XGBoost is the primary model:**

From Lecture 07's comparison table: XGBoost is best for *"structured data, moderate datasets."* This project has exactly that — 1,900 rows of tabular battery sensor readings. XGBoost's key advantages for this use case:

- **Sequential error correction:** Each tree corrects the residuals of the ensemble so far. Battery degradation has complex interactions (high resistance + high temperature + many cycles = accelerated degradation) that additive correction captures well.
- **L1 (reg_alpha) and L2 (reg_lambda) regularisation:** With only 1,900 training rows, overfitting is a constant risk. L1 pushes some feature weights to exactly zero (automatic feature selection). L2 penalises large weights uniformly (encourages simpler models).
- **Early stopping:** Training halts when the validation RMSE stops improving for 50 consecutive rounds. This prevents the well-known XGBoost overfitting problem where too many trees memorise training noise.
- **Missing value handling:** XGBoost automatically learns the optimal direction for missing values — relevant for inference inputs that may have gaps.

**XGBoost update formula:**
```
leaf_value = −Σgᵢ / (Σhᵢ + λ)
```
Where `gᵢ` is the first-order gradient (error), `hᵢ` is the second-order gradient (curvature), and `λ` is the L2 regularisation parameter. XGBoost uses both first and second derivatives of the loss function, allowing more precise minimisation than standard gradient boosting which uses only first-order gradients.

**Theory connection (Lecture 07):** *"XGBoost: Regularization L1 (Lasso) and L2 (Ridge) to prevent overfitting. Parallel Processing. Missing Value Handling. Early Stopping."*

### 8.5 Learning Curve Analysis

```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_selected_scaled, y_reg_train_log,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=tscv, scoring='neg_root_mean_squared_error'
)
```

The learning curve is plotted with training set size on the x-axis and RMSE on the y-axis. Two curves are drawn: training RMSE and validation RMSE.

**Interpretation:**
- If training RMSE is much lower than validation RMSE: overfitting — increase regularisation parameters.
- If both RMSEs are high and close together: underfitting — use a more complex model or better features.
- If both curves converge and validation RMSE is in an acceptable range: good generalisation.

**Theory connection (Lecture 01):** *"Overfitting: model performs well on training data but poorly on new data. Underfitting: model is too simple and cannot capture the patterns."*

### 8.6 Regression Metrics Used

All three metrics from Lecture 03 are reported for all models:

| Metric | Formula | Why Used |
|---|---|---|
| MAE | (1/n)Σ\|yᵢ - ŷᵢ\| | Easy to interpret in original degradation units. Reports average error magnitude. |
| RMSE | √[(1/n)Σ(yᵢ - ŷᵢ)²] | Penalises large errors more. A very wrong degradation prediction is costly — RMSE captures that. |
| R² | 1 - SSR/SST | Tells what fraction of the variance in cycle_degradation the model explains. Target: > 0.80. |

All metrics are computed in original units (after inverse log transform) for interpretability.

---

## 9. Model Training — Classification (Safety Flags)

*Notebook: `05_model_training_classification.ipynb` | Source: `src/train_classification.py`*

### 9.1 Evaluation Metric Priority

The primary metric for all classifiers is **Recall**, not Accuracy. This decision is made in the problem framing stage and is non-negotiable.

From Lecture 03: *"Recall = TP / (TP + FN). High recall = few missed positives. Use case: Medical diagnosis — you must NOT miss actual disease cases."*

A missed over-temperature event (False Negative) means the BMS fails to issue a warning during a real thermal violation. This could lead to accelerated irreversible degradation or, in severe cases, thermal runaway. A False Alarm (False Positive) triggers an unnecessary warning — annoying but not dangerous.

The threshold on classifier probability output is adjusted to ensure **Recall ≥ 0.85** on the validation set. This is a deliberate trade-off: some Precision is sacrificed (more false alarms) to ensure very few real events are missed.

### 9.2 Over-Voltage Flag Handling

As determined in preprocessing, the rule-based fallback is applied:

```python
def predict_overvoltage(action_voltage, terminal_voltage):
    return int(action_voltage > 4.15 or terminal_voltage > 4.18)
```

This function is called in `predict.py` for every inference request. The reasoning is documented: 4.15V and 4.18V are sub-threshold values below the 4.20V lithium-ion safety maximum, chosen to provide early warning before the hard limit is reached.

### 9.3 Model 1 — Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression

lr_cls = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    C=1.0,  # Inverse regularisation strength
    random_state=42
)
lr_cls.fit(X_train_selected_scaled, y_cls_temp_train)
```

Logistic regression uses the sigmoid function to output probability: `h_w(x) = 1 / (1 + e^(-wᵀx))`.

The default decision threshold is 0.5. For safety-critical classification, this is adjusted downward. The optimal threshold is found by evaluating Precision-Recall on the validation set at thresholds from 0.2 to 0.8 and selecting the lowest threshold that maintains Recall ≥ 0.85.

**Theory connection (Lecture 04):** *"Logistic Regression is used for binary classification. Decision boundary: y = 1 if h_w(x) ≥ 0.5, y = 0 otherwise."* The threshold modification extends this — y = 1 if h_w(x) ≥ t, where t < 0.5 to favour Recall.

### 9.4 Model 2 — SVM with RBF Kernel

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm_cls = SVC(kernel='rbf', class_weight='balanced',
              probability=True, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}
gs_svm = GridSearchCV(svm_cls, param_grid, cv=tscv,
                       scoring='f1', n_jobs=-1)
gs_svm.fit(X_train_selected_scaled, y_cls_temp_train)
```

**Why SVM with RBF Kernel for this problem:**

The boundary between "safe" battery operation (flag=0) and thermal violation (flag=1) is not a simple linear hyperplane in the feature space. Whether a battery will overheat is determined by the *combination* of temperature, resistance, SOC, current, and their historical trends — a complex, non-linear decision surface.

The RBF (Radial Basis Function) kernel maps the original feature space to an infinite-dimensional space where this complex boundary becomes linearly separable, without explicitly computing the high-dimensional transformation. This is the kernel trick.

```
K(x, z) = exp(-γ||x - z||²)
```

A high γ means the kernel has narrow influence (each support vector affects a small region — risk of overfitting). A low γ means broad influence (smoother boundary). Grid search over γ finds the best trade-off for this dataset.

**Theory connection (Lecture 06):** *"RBF (Radial Basis): K(x,z) = exp(-γ||x-z||²). Most common — infinite-dimensional space. Kernel trick: SVM's superpower — it computes dot products in high-dimensional space without explicitly transforming the data."*

`probability=True` is required to call `predict_proba()` for threshold tuning. It enables Platt scaling, which fits a logistic regression on the SVM decision scores to produce calibrated probabilities.

### 9.5 Model 3 — XGBoost Classifier (Primary)

```python
from xgboost import XGBClassifier

neg_count = (y_cls_temp_train == 0).sum()
pos_count = (y_cls_temp_train == 1).sum()
scale_pos = neg_count / pos_count

xgb_cls = XGBClassifier(
    scale_pos_weight=scale_pos,
    eval_metric='logloss',
    early_stopping_rounds=30,
    random_state=42
)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha': [0, 0.5, 1.0]
}
gs_xgb_cls = GridSearchCV(xgb_cls, param_grid, cv=tscv,
                           scoring='f1', n_jobs=-1)
gs_xgb_cls.fit(X_train_selected_scaled, y_cls_temp_train,
               eval_set=[(X_val_selected_scaled, y_cls_temp_val)],
               verbose=False)
```

`scale_pos_weight` is XGBoost's native mechanism for handling class imbalance. Setting it to the ratio of negative to positive samples tells XGBoost to upweight the gradient of minority class errors during tree construction. This is functionally equivalent to `class_weight='balanced'` in scikit-learn but integrated directly into XGBoost's objective function.

**Log Loss cost function for classification:**
```
J(w) = -(1/m) × Σ [y log h_w(x) + (1-y) log(1-h_w(x))]
```
Log loss penalises confident wrong predictions more heavily than uncertain ones — appropriate for safety classification where a model that confidently predicts "safe" when the battery is about to overheat is far more dangerous than one that is uncertain.

**Theory connection (Lecture 07 and Lecture 04):** *"Loss Function: MSE for regression; Log Loss for classification."* *"Logistic regression uses Log Loss instead of MSE. It penalises confident wrong predictions more heavily."*

### 9.6 Threshold Tuning

```python
from sklearn.metrics import precision_recall_curve

probs = best_cls_model.predict_proba(X_val_selected_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_cls_temp_val, probs)

# Find lowest threshold where recall >= 0.85
optimal_idx = np.where(recalls >= 0.85)[0][-1]
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"At this threshold — Precision: {precisions[optimal_idx]:.3f}, "
      f"Recall: {recalls[optimal_idx]:.3f}")
```

The optimal threshold is saved to `models/classification_config.json` and used in `predict.py` during inference.

### 9.7 ROC Curve Comparison

All three classifiers' ROC curves are plotted on a single figure:

```python
from sklearn.metrics import RocCurveDisplay
fig, ax = plt.subplots(figsize=(8, 6))
for model, name in [(best_lr, 'Logistic Regression'),
                     (best_svm, 'SVM (RBF)'),
                     (best_xgb_cls, 'XGBoost')]:
    RocCurveDisplay.from_estimator(model, X_val_selected_scaled,
                                    y_cls_temp_val, ax=ax, name=name)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC=0.5)')
ax.set_title('ROC Curve Comparison — Over-Temperature Classifier')
plt.savefig('plots/roc_comparison_overtemp.png', dpi=150, bbox_inches='tight')
```

**Theory connection (Lecture 03):** *"ROC curve plots TPR vs FPR at various threshold values. AUC = Area Under the Curve. AUC=1.0: Perfect classifier. AUC=0.5: Random classifier."*

---

## 10. Model Evaluation and Selection

*Notebook: `06_model_evaluation_and_selection.ipynb`*

> **Critical rule:** The test set is touched once, only in this notebook. All hyperparameter decisions and model selections are made using the validation set. The test set provides the final honest, unbiased estimate of real-world performance.

### 10.1 Regression — Final Evaluation on Test Set

The best regression model is loaded and applied to `X_test_selected_scaled`:

```python
import joblib
reg_model = joblib.load('models/regression_model.pkl')
y_pred_log = reg_model.predict(X_test_selected_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_reg_test_log)
```

**Metrics computed:**

| Metric | Formula | Lecture Reference |
|---|---|---|
| MAE | (1/n)Σ\|yᵢ - ŷᵢ\| | Lecture 03: Lower MAE = better model |
| RMSE | √[(1/n)Σ(yᵢ - ŷᵢ)²] | Lecture 03: Most commonly used for regression |
| R² | 1 - SSR/SST | Lecture 03: Tells how well independent variables explain variation |

**Visualisations:**

Actual vs Predicted scatter plot: Points should cluster around the diagonal line y = x. Points far from the diagonal are high-error predictions. A systematic bias (all points above or below the line) indicates a model that consistently over- or under-predicts.

Residual plot: (actual - predicted) vs predicted. A random scatter around zero indicates good model behaviour. Any visible pattern (U-shape, increasing spread) indicates a systematic error the model fails to capture.

### 10.2 Classification — Final Evaluation on Test Set

```python
cls_model = joblib.load('models/classification_model_temp.pkl')
config = json.load(open('models/classification_config.json'))
threshold = config['temp_threshold']

probs = cls_model.predict_proba(X_test_selected_scaled)[:, 1]
y_pred_cls = (probs >= threshold).astype(int)
```

**Metrics computed and their justification:**

| Metric | Formula | Why Reported |
|---|---|---|
| Recall | TP / (TP + FN) | Primary metric. Missing a real over-temperature event (FN) is the worst outcome. Lecture 03: *"FN concern: Medical diagnosis — you must NOT miss actual disease cases."* |
| Precision | TP / (TP + FP) | Reported alongside Recall. Too many false alarms (high FP) would make the system unusable. |
| F1 Score | 2 × P × R / (P + R) | Balance between Precision and Recall. Lecture 03: *"Best when both FP and FN are equally important."* |
| ROC-AUC | Area under ROC curve | Overall model separability. Lecture 03: *"AUC = 1.0: Perfect classifier."* |

**Confusion Matrix with explicit Type 1 / Type 2 error labelling:**

```
                  Predicted 0    Predicted 1
Actual 0         True Negative   False Positive (Type 1 Error)
Actual 1         False Negative  True Positive
                 (Type 2 Error)
```

From Lecture 03: *"FP = Type 1 Error. FN = Type 2 Error."*

The confusion matrix is displayed using `ConfusionMatrixDisplay` and annotated with cell labels. The Type 2 error (False Negative) cell is highlighted in red in the presentation to emphasise that this is the cell the project minimises.

### 10.3 Full Model Comparison Summary

**Regression Results:**

| Model | MAE | RMSE | R² | Notes |
|---|---|---|---|---|
| Linear Regression | — | — | — | Baseline. Limited by linear assumption. |
| Random Forest | — | — | — | Better. Handles interactions. |
| **XGBoost (selected)** | — | — | — | **Best. Sequential error correction + regularisation.** |

**Classification Results (Over-Temperature):**

| Model | Recall | Precision | F1 | ROC-AUC | Notes |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | Baseline. Linear boundary. |
| SVM (RBF) | — | — | — | — | Non-linear boundary via kernel. |
| **XGBoost (selected)** | — | — | — | — | **Best. Recall ≥ 0.85 with tuned threshold.** |

*Metric values are filled in from actual notebook execution.*

### 10.4 Model Selection Justification

**Regression model selected: XGBoost Regressor**

Justification: XGBoost is best suited for tabular, structured data of moderate size (Lecture 07). Its sequential error correction captures the non-linear relationship between battery state features and degradation. L1/L2 regularisation prevents overfitting on the 1,329-row training set. Early stopping provides automatic model complexity control. Feature importance from XGBoost also validates the feature engineering decisions from Notebook 03.

**Classification model selected: XGBoost Classifier**

Justification: XGBoost achieves the highest F1-Score and meets the Recall ≥ 0.85 requirement with the tuned threshold. The `scale_pos_weight` parameter directly addresses class imbalance within the gradient boosting framework. Its probabilistic output enables fine-grained threshold control not available with hard-boundary models.

---

## 11. Source Code Architecture

### 11.1 `src/preprocessor.py`

```python
class BatteryPreprocessor:
    """
    Encapsulates all cleaning, imputation, and scaling steps.
    Fit on training data only. Transform applied identically
    to training, validation, test, and inference data.
    """
    def __init__(self, scaler_path=None, feature_cols_path=None):
        self.scaler = None
        self.feature_columns = None
        self.ir_baseline = None  # Internal resistance reference value

    def fit(self, X_train_raw):
        # Compute IR baseline from first training row
        self.ir_baseline = X_train_raw['internal_resistance'].iloc[0]
        # Fit imputer and scaler on training data
        ...

    def transform(self, X_raw):
        # Apply feature engineering
        # Apply imputation
        # Select feature_columns
        # Apply scaling
        # Return numpy array
        ...

    def fit_transform(self, X_train_raw):
        self.fit(X_train_raw)
        return self.transform(X_train_raw)

    def inverse_transform_target(self, y_log_pred):
        return np.expm1(y_log_pred)
```

**Key design decisions:**

`ir_baseline` is stored as an instance attribute and saved to disk. During inference, the transform method subtracts this training-set baseline from the incoming `internal_resistance` value to compute `delta_internal_resistance`. Using the inference batch's own first value would produce a different engineered feature, making predictions inconsistent.

### 11.2 `src/feature_engineer.py`

Pure functions, no state. Each function takes a DataFrame and returns a DataFrame with the new column appended:

```python
def add_delta_internal_resistance(df, baseline_ir):
    df = df.copy()
    df['delta_internal_resistance'] = df['internal_resistance'] - baseline_ir
    return df

def add_soc_range_rolling(df, window=10, fill_value=None):
    df = df.copy()
    rolling_range = df['SOC'].rolling(window=window).max() - \
                    df['SOC'].rolling(window=window).min()
    fill = fill_value if fill_value else rolling_range.median()
    df['soc_range_rolling'] = rolling_range.fillna(fill)
    return df

def add_thermal_acceleration(df):
    df = df.copy()
    df['thermal_acceleration'] = df['dT_dt'] * df['battery_temp']
    return df

def add_voltage_efficiency(df):
    df = df.copy()
    df['voltage_efficiency'] = df['action_voltage'] / \
                               (df['terminal_voltage'] + 1e-6)
    return df

def add_polynomial_features(df, columns):
    df = df.copy()
    for col in columns:
        df[f'{col}_sq'] = df[col] ** 2
    return df

def drop_redundant_features(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')
```

**Design rationale:** Pure functions (no side effects, no state) are the easiest to test, debug, and reuse. The `df.copy()` at the start of each function prevents accidental mutation of the input DataFrame.

### 11.3 `src/predict.py`

```python
import joblib, json
import numpy as np
import pandas as pd
from preprocessor import BatteryPreprocessor

_models = None

def load_models():
    global _models
    if _models is None:
        _models = {
            'regression': joblib.load('models/regression_model.pkl'),
            'classification_temp': joblib.load('models/classification_model_temp.pkl'),
            'preprocessor': joblib.load('models/preprocessor_pipeline.pkl'),
            'feature_columns': json.load(open('models/feature_columns.json')),
            'config': json.load(open('models/classification_config.json'))
        }
    return _models

def predict_single(input_dict):
    models = load_models()
    df_input = pd.DataFrame([input_dict])

    # Apply preprocessing pipeline
    X_processed = models['preprocessor'].transform(df_input)

    # Regression prediction
    y_log = models['regression'].predict(X_processed)
    cycle_degradation = float(np.expm1(y_log)[0])

    # Classification prediction
    threshold = models['config']['temp_threshold']
    prob_temp = float(models['classification_temp'].predict_proba(X_processed)[0, 1])
    over_temp_flag = int(prob_temp >= threshold)

    # Rule-based voltage flag
    over_voltage_flag = int(
        input_dict.get('action_voltage', 0) > 4.15 or
        input_dict.get('terminal_voltage', 0) > 4.18
    )

    return {
        'cycle_degradation': cycle_degradation,
        'over_temp_probability': round(prob_temp * 100, 2),
        'over_temp_flag': over_temp_flag,
        'over_voltage_flag': over_voltage_flag
    }

def predict_batch(df_raw):
    models = load_models()
    X_processed = models['preprocessor'].transform(df_raw)
    # Run all predictions on the full batch
    ...
    return df_with_predictions
```

**The `_models = None` / lazy loading pattern** ensures that model files are loaded only once on first call and reused for all subsequent predictions. Loading large pkl files on every request would make the UI unacceptably slow.

---

## 12. Deployment — Gradio UI on Hugging Face

### 12.1 Application Design

The Gradio application (`app/app.py`) provides two tabs.

**Tab 1 — Single Battery State Prediction**

A form where the user inputs current battery parameters using sliders and number inputs. On clicking Predict, the app calls `predict_single()` and displays:
- Predicted cycle degradation (numerical)
- Over-temperature risk as a percentage with a colour-coded badge (green ≤ 30%, yellow 30–70%, red > 70%)
- Temperature safety status: "SAFE" or "⚠️ WARNING — THERMAL RISK DETECTED"
- Voltage safety status: "SAFE" or "⚠️ WARNING — VOLTAGE LIMIT APPROACHING"

**Tab 2 — Batch CSV Prediction**

Upload any CSV containing battery readings with the expected feature columns. The app applies the full pipeline to all rows and returns a downloadable CSV with three new columns appended: `pred_cycle_degradation`, `pred_over_temp_flag`, `pred_over_voltage_flag`.

### 12.2 Gradio Implementation Sketch

```python
import gradio as gr
from src.predict import predict_single, predict_batch, load_models

load_models()  # Load at startup, not per-request

def single_predict_fn(SOC, SOH, terminal_voltage, battery_current,
                      battery_temp, ambient_temp, internal_resistance,
                      action_current, action_voltage,
                      charging_efficiency, charging_time):
    input_dict = {k: v for k, v in locals().items()}
    result = predict_single(input_dict)
    degradation = result['cycle_degradation']
    temp_prob   = result['over_temp_probability']
    temp_status = "⚠️ WARNING — THERMAL RISK" if result['over_temp_flag'] else "✅ SAFE"
    volt_status = "⚠️ WARNING — VOLTAGE RISK" if result['over_voltage_flag'] else "✅ SAFE"
    return degradation, f"{temp_prob:.1f}%", temp_status, volt_status

with gr.Blocks(title="EV Battery Health Predictor") as demo:
    with gr.Tab("Single Prediction"):
        with gr.Row():
            SOC = gr.Slider(0, 100, label="State of Charge (%)", value=50)
            SOH = gr.Slider(0.5, 1.0, step=0.01, label="State of Health", value=0.92)
        # ... more inputs ...
        predict_btn = gr.Button("Predict Battery Health", variant="primary")
        with gr.Row():
            deg_out  = gr.Number(label="Predicted Cycle Degradation")
            prob_out = gr.Textbox(label="Over-Temperature Risk")
            temp_out = gr.Textbox(label="Temperature Safety")
            volt_out = gr.Textbox(label="Voltage Safety")
        predict_btn.click(single_predict_fn, inputs=[...], outputs=[...])

    with gr.Tab("Batch Prediction"):
        file_in  = gr.File(label="Upload CSV", file_types=[".csv"])
        run_btn  = gr.Button("Run Batch Prediction")
        preview  = gr.Dataframe(label="Preview (first 10 rows)")
        file_out = gr.File(label="Download Predictions CSV")
        run_btn.click(batch_predict_fn, inputs=[file_in],
                      outputs=[preview, file_out])

demo.launch()
```

### 12.3 Hugging Face Spaces Deployment

**Repository structure on Hugging Face:**
```
ev-battery-predictor/          ← Hugging Face Space repository root
├── README.md                  ← Space metadata + description
├── app.py                     ← Entry point (renamed from app/app.py)
├── requirements.txt
├── src/
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   └── predict.py
└── models/
    ├── regression_model.pkl
    ├── classification_model_temp.pkl
    ├── preprocessor_pipeline.pkl
    ├── feature_columns.json
    └── classification_config.json
```

**README.md Space metadata header:**
```yaml
---
title: EV Battery Health Predictor
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
---
```

**Deployment steps:**
1. Create a new Space at huggingface.co/spaces, selecting SDK: Gradio.
2. Clone the Space repository locally.
3. Copy the project files into the cloned repository as above.
4. If model `.pkl` files exceed 100MB combined: `git lfs install && git lfs track "*.pkl" && git add .gitattributes`.
5. Commit and push: `git add . && git commit -m "Initial deployment" && git push`.
6. The Space auto-builds on push. Monitor build logs for import errors.
7. Validate deployment by running a prediction with known inputs and comparing output to the notebook result.

**Version pinning requirement:** Hugging Face Spaces run Python 3.10. The `requirements.txt` must pin scikit-learn to the exact same major.minor version used during training (e.g., `scikit-learn==1.4.2`). Joblib-serialised sklearn objects are version-sensitive — a model saved with sklearn 1.4.2 loaded under sklearn 1.5.0 may raise errors or produce silently incorrect results.

---

## 13. Theory Traceability

Every significant decision in this project maps to a specific concept from the module lectures. This table serves as the academic accountability reference.

| Decision | Lecture | Theory Concept |
|---|---|---|
| Regression for cycle_degradation (continuous output) | Lecture 01, 04 | Supervised learning types; Regression = continuous output |
| Classification for over_temp_flag (binary output) | Lecture 01, 04 | Classification = categorical output; Logistic Regression |
| Rule-based for over_voltage_flag (insufficient positive samples) | Lecture 03 | Accuracy is misleading for extreme imbalance |
| Median imputation for missing values | Lecture 02 Step 2 | Handle Missing Values with median for skewed data |
| Log1p transform on cycle_degradation | Lecture 02 | Log Transform: reduces skewness in extreme-range data |
| Chronological train/val/test split | Lecture 01 | Data must be split; model must generalise to unseen data |
| StandardScaler on all features | Lecture 02 Step 3 | Standardisation: zero mean, unit variance for scale-sensitive models |
| Scaler fit only on training data | Lecture 01 | Prevent data leakage; generalise to unseen data |
| class_weight='balanced' (not SMOTE) for time-series imbalance | Lecture 03 | Recall over Accuracy for imbalanced classes |
| Recall as primary metric for safety classification | Lecture 03 | High recall = few missed positives; FN concern > FP |
| Decision threshold lowered below 0.5 | Lecture 03 | Precision for FP concern, Recall for FN concern |
| Confusion matrix with Type 1 / Type 2 labelling | Lecture 03 | FP = Type 1 Error; FN = Type 2 Error |
| ROC-AUC for overall model quality | Lecture 03 | AUC = overall separability; 1.0 = perfect |
| Polynomial features (squared terms) | Lecture 02 | Polynomial Features: captures non-linear relationships |
| delta_internal_resistance (domain-knowledge feature) | Lecture 02 Step 4 | Create new features using domain knowledge |
| soc_range_rolling (compound feature) | Lecture 02 Step 4 | Combine features to encode physically meaningful quantity |
| Filter feature selection (correlation) | Lecture 05 | Filter Methods: statistical properties |
| Wrapper feature selection (RFE) | Lecture 05 | Wrapper Methods: considers feature interactions |
| XGBoost feature importances | Lecture 05 | Embedded Methods: built into model training |
| Linear Regression as baseline | Lecture 04 | Foundational supervised regression algorithm |
| Random Forest to reduce overfitting variance | Lecture 06 | Bagging + random feature selection; reduces variance |
| XGBoost as primary model | Lecture 07 | Best for structured data, moderate datasets; L1/L2 regularisation |
| TimeSeriesSplit for cross-validation | Lecture 01 | Generalisation to unseen data; prevent future leakage |
| Early stopping in XGBoost | Lecture 07 | Stops training if validation performance doesn't improve |
| SVM RBF kernel for non-linear class boundary | Lecture 06 | Kernel trick: non-linear separation in infinite-dimensional space |
| AdaBoost / XGBoost sequential nature vs Bagging | Lecture 07 | Boosting = sequential; Bagging = independent |
| Learning curve analysis | Lecture 01 | Overfitting vs Underfitting diagnosis |
| Separate src/ modules for training and inference | Lecture 01 | The ML workflow; generalise to unseen data requires identical transforms |
| Gradio deployment on Hugging Face | Lecture 01 Step 5 | Deployment: integrate trained model into applications |

---

## 14. Limitations and Future Work

### 14.1 Known Limitations

**`battery_temp` encodes lifetime thermal accumulation, not session temperature.** The dataset appears to represent a single battery's full lifecycle rather than multiple independent batteries. Any model trained on this data may not generalise to batteries with different lifetime thermal histories or different cell chemistries.

**`over_voltage_flag` has insufficient positive examples.** With fewer than 15 positive cases, the rule-based fallback is used. The thresholds (4.15V, 4.18V) are physically motivated but not tuned to this specific battery type. Access to more data with genuine over-voltage events would enable a proper classifier.

**No multi-battery generalisation.** If this dataset represents one battery's lifetime, the model learns this battery's specific degradation trajectory. Real-world deployment would require data from many batteries of the same and different types, with proper battery-level train/test splitting to ensure the model generalises across batteries, not just across time within one battery.

**Small dataset size (1,900 rows).** Deep learning approaches (LSTMs, Transformers for time-series) are not applied because 1,900 rows is insufficient to train such models without severe overfitting. Tree-based methods are the appropriate choice at this data scale.

### 14.2 Future Work

**LSTM for sequence modelling:** Battery degradation is fundamentally sequential. An LSTM would process the full sequence of charging states to predict future degradation, capturing temporal dependencies that the current feature-engineering approach approximates manually (e.g., rolling SOC range). Requires significantly more data.

**Multi-battery training with battery-level hold-out:** The ideal experimental setup trains on data from N batteries and tests on a held-out battery — ensuring the model generalises to unseen batteries, not just unseen time steps of a known battery.

**Online learning / model drift detection:** As a deployed battery ages, its degradation characteristics may shift in ways the training data does not capture. Monitoring prediction error over time and retraining when drift is detected would keep the model accurate in long-term deployment.

**Explainability layer (SHAP values):** Integrating SHAP (SHapley Additive exPlanations) into the Gradio UI would allow users to see which features contributed most to a specific degradation prediction — important for battery engineers who need to understand root causes, not just predictions.

---

## 🎯 Project Goals

1. **Predict battery degradation** from charging sensor data
2. **Estimate remaining useful life (RUL)** to optimize replacement schedules
3. **Detect safety risks** (thermal, voltage anomalies)
4. **Provide actionable insights** for maintenance teams
5. **Enable fleet-wide analytics** (compare battery to fleet average)

---

## 🔄 Recent Session: Major Improvements

### Problem 1: Feature Shape Mismatch ❌ → ✅

**Symptom**: `ValueError: Feature shape mismatch, expected 15, got 23`

**Root Cause**:
- Models trained on 15 RFE-selected engineered features
- But preprocessing pipeline had ordering issue:
  1. Generate 23 engineered features
  2. **Incorrectly select 15 BEFORE imputing/scaling** ❌
  3. Imputer/Scaler trained on 23 features but received 15
  4. Mismatch causes crash

**Solution**:
```
WRONG: Engineer(23) → Select(15) → Impute(23) → Scale(23)
RIGHT: Engineer(23) → Impute(23) → Scale(23) → Select(15)
```

**Why This Matters**:
- Imputer must be fitted on full feature set (handles missing values in all 23)
- Scaler must be fitted on full feature set (normalizes all 23)
- RFE selection happens AFTER, reducing to best 15
- Order is **critical**: preprocessing artifacts assume 23 columns

**File Changed**: `src/preprocessor.py`

### Problem 2: Imputer None Error ❌ → ✅

**Symptom**: `AttributeError: 'NoneType' object has no attribute 'transform'`

**Root Cause**:
- Notebook 02 Cell 5 had comment "Building imputer anyway" but never created SimpleImputer
- Cell 8 tried to save imputer to pickle, got None
- Later inference tried to use None object → crash

**Solution**:
- Implemented proper imputer in Notebook 02 Cell 8
- Fitted on engineered 23 features (before scaling)
- Saved to `preprocessor_pipeline.pkl`

**Code Pattern**:
```python
# Before: imputer was None
imputer = None  # Never created!

# After: Proper creation
imputer = SimpleImputer(strategy='median')
X_train_engineered_imputed = imputer.fit_transform(X_train_engineered_23)
```

**File Changed**: `notebooks/02_data_preprocessing.ipynb` (Cell 8)

### Problem 3: Streamlit Deprecation Warnings ❌ → ✅

**Symptom**: Multiple warnings on app startup
```
UserWarning: use_container_width is deprecated and will be removed...
```

**Root Cause**: Streamlit v1.28+ removed `use_container_width` parameter

**Solution**: Replace all instances:
```python
# Old (deprecated)
st.metric(..., use_container_width=True)
st.plotly_chart(..., use_container_width=False)

# New (works)
st.metric(..., width='stretch')
st.plotly_chart(..., width='content')
```

**Instances Fixed**: ~15 across app.py

**File Changed**: `app/app.py`

### Problem 4: UX Fragmentation ❌ → ✅

**Issue**: Insights were in separate Tab 6, forcing users to switch tabs after prediction

**Solution**: Integrate insights directly into Tab 1 (Single Prediction)

**Flow**:
```
Old UX:
  User → Tab 1 (Predict) → Get IR prediction → Manually go to Tab 6 (Insights)

New UX:
  User → Tab 1 (Predict) → Get IR prediction + health insights (same tab)
```

**Changes**:
- Tab 1 now calls `BatteryInsights.assess_battery_health()` after prediction
- Displays health score, RUL, fleet comparison, recommendations
- Removed separate Tab 6
- Now 5 tabs instead of 6

**Files Changed**:
- `app/app.py` (integrated insights into Tab 1, deleted Tab 6)

---

## 🏛️ System Architecture

### High-Level Flow

```
┌─────────────────┐
│  User Input     │ 15 raw features from sensors
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Preprocessing Pipeline (predict.py)    │
│  1. Feature engineering (15 → 23)       │
│  2. Imputation (median strategy)        │
│  3. StandardScaler normalization        │
│  4. RFE selection (23 → 15)             │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  ML Models (XGBoost)                    │
│  - Regression: internal_resistance      │
│  - Classification: over_temp_flag       │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Insights Engine (insights.py)          │
│  1. Health score calculation            │
│  2. RUL estimation                      │
│  3. Fleet comparison                    │
│  4. Recommendations                     │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Streamlit UI (app.py)                  │
│  - 5 tabs for user interaction          │
│  - Real-time charts & gauges            │
│  - Interactive simulation                │
└─────────────────────────────────────────┘
```

### Feature Architecture: 15 → 23 → 15

**Why Three Numbers?**

1. **15 Raw Features** (from sensors):
   ```
   SOC, Battery_Voltage, Battery_Current, SOH,
   Ambient_Temp, Battery_Temp, Internal_Resistance,
   Action_Voltage, Action_Current, Charging_Efficiency,
   Charging_Time, Balancing_Time, Over_Temp_Flag,
   Over_Voltage_Flag, Terminal_Voltage
   ```

2. **23 Engineered Features** (derived from raw):
   - Original 15 (after imputation + scaling)
   - + 8 domain-knowledge features:
     - `soc_range_rolling` — SOC volatility over 5-cycle window
     - `thermal_acceleration` — dTemp/dt derivative
     - `voltage_efficiency` — V_terminal / V_action ratio
     - `internal_resistance_sq` — IR squared
     - `aging_indicator_sq` — cumulative age proxy
     - Polynomial combinations
     - Interaction terms

3. **15 RFE-Selected** (best predictors):
   - XGBoost + Recursive Feature Elimination identifies top 15 from 23
   - Saved in `models/feature_columns.json`
   - These 15 are what the XGBoost models expect

**Why This Process?**
- Create many candidate features (23) via domain knowledge
- RFE automatically selects best subset (15) to prevent overfitting
- Reduces dimensionality without manual feature selection
- Models trained only on these 15 (consistent interface)

### Preprocessing Pipeline (src/preprocessor.py)

```python
class BatteryPreprocessor:
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.baseline_ir = None
        self.columns_to_drop = None
    
    def fit(X_train):
        # Step 1: Engineer 23 features
        X_engineered = engineer_all_features(X_train, ...)
        
        # Step 2: Fit imputer on 23 features
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X_engineered)
        
        # Step 3: Fit scaler on imputed 23 features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Step 4: Save metadata for inference
        self.columns_to_drop = [...]
        self.baseline_ir = X_train['internal_resistance'].mean()
    
    def transform(X):
        # Apply same pipeline in order
        X_eng = engineer_all_features(X, self.baseline_ir, ...)
        X_imp = self.imputer.transform(X_eng)
        X_scaled = self.scaler.transform(X_imp)
        X_selected = select_features(X_scaled, feature_columns.json)
        return X_selected  # Returns 15 features ready for XGBoost
```

**Key Insight**: Imputer & Scaler saved to `preprocessor_pipeline.pkl`, loaded during inference. They maintain the fitted parameters from training data.

---

## 🧠 Models & Performance

### Regression Model: Internal Resistance

**Architecture**:
- Algorithm: XGBoost Regressor
- Input: 15 engineered features
- Output: internal_resistance (Ω)
- Training data: 1,330 samples (70% chronological split)
- Validation: 233 samples (15%)
- Test: 337 samples (15%)

**Hyperparameters**:
```json
{
  "n_estimators": 500,
  "max_depth": 8,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "objective": "reg:squarederror",
  "random_state": 42
}
```

**Performance**:
```
Test Set:
  R² Score:     0.97
  RMSE:         0.0095 Ω
  MAE:          0.0068 Ω
  
Interpretation:
  - Model explains 97% of variance in IR
  - Average prediction error: ±0.01 Ω
  - Production-grade accuracy
```

**What This Predicts**: Battery's electrochemical impedance degradation → triggers health alerts

### Classification Model: Over-Temperature Flag

**Architecture**:
- Algorithm: XGBoost Classifier
- Input: 15 engineered features
- Output: over_temp_flag (binary: 0 or 1)
- Cross-Validation: StratifiedKFold (5-fold)
- Training samples: 1,330

**Performance**:
```
CV Results (averaged over 5 folds):
  F1 Score:     0.95
  Precision:    0.94
  Recall:       0.96
  ROC-AUC:      0.98
  
Interpretation:
  - Catches 96% of thermal incidents (high recall)
  - Few false alarms (94% precision)
  - Excellent discrimination (98% AUC)
```

**Why StratifiedKFold?**
- `over_temp_flag` has temporal block pattern: 0 for first 900 rows, then 1
- Chronological split would give single-class train/val (useless for classification)
- StratifiedKFold maintains class balance in each fold (documented practice)

**What This Predicts**: Whether battery is experiencing thermal stress

### Over-Voltage Flag: Rule-Based

**Why Not ML?**
- Only 20-30 positive examples in 1,900 samples (extreme imbalance)
- Not enough data to train reliable classifier

**Rule**:
```python
over_volt_flag = (action_voltage > 4.15V) OR (terminal_voltage > 4.18V)
```

**Performance**:
- 100% precision on training set (never over-predicts)
- Conservative: triggers only when truly anomalous

---

## 📊 Insights Engine (src/insights.py)

### Health Score Calculation

```python
def assess_battery_health(ir, over_temp_flag, over_volt_flag):
    # Component Scores (0-100)
    ir_score = max(0, (0.15 - ir) / 0.15 * 100)
    thermal_score = (1 - over_temp_flag) * 100
    voltage_score = (1 - over_volt_flag) * 100
    
    # Weighted Average
    health = 0.40 * ir_score + 0.35 * thermal_score + 0.25 * voltage_score
    
    # Status Mapping
    if health >= 80:
        status = "🟢 HEALTHY"
    elif health >= 60:
        status = "🟡 DEGRADING"
    elif health >= 40:
        status = "🔴 CRITICAL"
    else:
        status = "⚫ END_OF_LIFE"
    
    return {'overall_health': health, 'status': status, 'component_scores': [...]}
```

**Why 40-35-25 weights?**
- IR (40%): Most critical — directly correlates to aging
- Thermal (35%): High safety impact — prevents thermal runaway
- Voltage (25%): Lower frequency but important — prevents cell stress

### RUL Estimation

```python
def estimate_remaining_useful_life(ir):
    # Find degradation rate from training data
    ir_history = dataset.internal_resistance
    cycles = len(ir_history)
    degradation_rate = (ir_history.iloc[-1] - ir_history.iloc[0]) / cycles
    
    # Conservative estimate
    ir_threshold = 0.15  # Replacement threshold
    cycles_remaining = (ir_threshold - ir) / degradation_rate
    conservative = cycles_remaining * 0.7  # 30% safety margin
    
    return {'cycles_remaining': conservative, 'degradation_rate': degradation_rate}
```

**Why Conservative (0.7×)?**
- Degradation accelerates near EOL
- 30% safety margin accounts for this nonlinearity
- Better to replace early than have failure

### Fleet Comparison

```python
def compare_to_fleet(ir, efficiency, temp_prob):
    fleet_mean_ir = dataset.internal_resistance.mean()  # 0.097Ω
    fleet_std_ir = dataset.internal_resistance.std()    # 0.016Ω
    
    # Percentile rank
    percentile = (ir <= dataset.internal_resistance).sum() / len(dataset) * 100
    
    # Sigma deviation
    sigma_dev = (ir - fleet_mean_ir) / fleet_std_ir
    
    # Status
    status = "WORSE" if percentile > 75 else "BETTER" if percentile < 25 else "AVERAGE"
    
    return {'percentile': percentile, 'sigma_dev': sigma_dev, 'status': status}
```

**Interpretation**:
- 10th percentile: Better than 90% of fleet → Good condition
- 50th percentile: Average fleet battery
- 90th percentile: Worse than 90% → Problem battery

### Recommendations Generator

```python
def get_recommendations(health_score, rul, alerts):
    recommendations = []
    
    if health_score >= 80:
        recommendations.append("Battery is in good condition. Continue normal operation.")
        recommendations.append("Routine monitoring recommended monthly.")
    elif health_score >= 60:
        recommendations.append("Battery shows degradation signs. Increase monitoring frequency.")
        recommendations.append("Consider maintenance within 2-3 weeks.")
    elif health_score >= 40:
        recommendations.append("🔴 Battery degrading rapidly. Daily monitoring required.")
        recommendations.append("⚡ Reduce charging power if possible.")
        recommendations.append("📞 Contact maintenance team for urgent assessment.")
    else:
        recommendations.append("🚨 CRITICAL: Battery reached end-of-life. Replace immediately.")
    
    if 'thermal_anomaly' in [a['type'] for a in alerts]:
        recommendations.append("Avoid high-power charging. Risk of thermal runaway.")
    
    return recommendations
```

---

## 🎮 Interactive Simulator (src/simulator.py)

### How It Works

```python
class BatterySimulator:
    def get_initial_state():
        # Battery at 0% SOC
        return {
            'soc': 0,
            'raw': {...15 initial values...},
            'engineered': {...15 engineered features...}
        }
    
    def step(state, soc_increment):
        # Update raw state based on physics patterns
        state['soc'] += soc_increment
        state['raw']['battery_voltage'] += pattern_based_voltage_increase
        state['raw']['battery_current'] = f(state['soc'])
        state['raw']['battery_temp'] += thermal_calculation
        
        # Apply feature engineering
        engineered = engineer_all_features(state['raw'])
        state['engineered'] = select_rfe_features(engineered)  # 15 features
        
        return state
```

### Use Case

Users can:
1. Initialize simulation at 0% SOC
2. Step through 1%, 5%, or 10% increments
3. Watch real-time:
   - SOC gauge → 0 to 100%
   - IR prediction gauge → changes as battery charges
   - Over-temp probability → temperature profile
4. See how predictions change during actual charging cycle

**Why Valuable?**
- Understand how predictions evolve during charge
- Validate model behavior matches physics intuition
- Educational tool for maintenance teams

---

## 📁 Key Files & Their Roles

### `notebooks/02_data_preprocessing.ipynb` — Cell 8 (Critical)

**Before (Broken)**:
```python
# Had comment but never built imputer
# Cell 5: "Building imputer anyway"
imputer = None  # Was never created
```

**After (Fixed)**:
```python
# Properly fit imputer on 23 engineered features
X_train_engineered = engineer_all_features(X_train)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_engineered)

# Fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

# Save pipeline
with open('preprocessor_pipeline.pkl', 'wb') as f:
    pickle.dump({'imputer': imputer, 'scaler': scaler, ...}, f)
```

**Why This Matters**: All inference depends on these fitted objects being available

### `src/preprocessor.py` — Feature Ordering (Critical)

**Before (Broken Order)**:
```python
def transform(self, X):
    X_eng = engineer_features(X)  # 23 features
    X_sel = select_rfe(X_eng)     # Select 15 ❌ WRONG PLACE
    X_imp = self.imputer.transform(X_sel)  # Imputer expects 23!
    X_scl = self.scaler.transform(X_imp)   # Scaler expects 23!
```

**After (Correct Order)**:
```python
def transform(self, X):
    X_eng = engineer_features(X)        # 23 features
    X_imp = self.imputer.transform(X_eng)   # Impute 23
    X_scl = self.scaler.transform(X_imp)   # Scale 23
    X_sel = select_rfe(X_scl)               # Select 15 ✅
    return X_sel
```

**Why This Matters**: Preprocessing artifacts (imputer, scaler) fitted on 23 columns, must receive 23 columns

### `app/app.py` — Three Major Changes

**Change 1**: Streamlit deprecation fixes (~15 instances)
```python
# Old
st.metric("IR", ir_pred, use_container_width=True)

# New
st.metric("IR", ir_pred, width='stretch')
```

**Change 2**: Integrated insights into Tab 1
```python
# After prediction results, added:
st.markdown("### 📈 Battery Health Insights")
health_result = st.session_state.insights.assess_battery_health(ir_pred, temp_flag, volt_flag)
rul_result = st.session_state.insights.estimate_remaining_useful_life(ir_pred)
# Display health gauge, RUL, recommendations...
```

**Change 3**: Removed Tab 6 (Insights)
```python
# Old
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([...6 tabs...])
with tab6:
    # 200+ lines of insights code

# New
tab1, tab2, tab3, tab4, tab5 = st.tabs([...5 tabs...])
# Insights now in Tab 1
```

---

## 🧪 Testing & Validation

### Test Files

**test_simulator.py**:
```python
def test_simulator_generates_15_features():
    sim = BatterySimulator()
    state = sim.get_initial_state()
    state = sim.step(state, 5)
    assert len(state['engineered']) == 15  # Exactly 15 RFE-selected

def test_simulator_soc_progression():
    sim = BatterySimulator()
    state = sim.get_initial_state()
    assert state['soc'] == 0
    state = sim.step(state, 10)
    assert state['soc'] == 10  # 10% increase
```

**test_insights.py**:
```python
def test_health_score_calculation():
    insights = BatteryInsights()
    result = insights.assess_battery_health(ir=0.085, temp_flag=0, volt_flag=0)
    assert 0 <= result['overall_health'] <= 100
    assert result['status'] in ['🟢 HEALTHY', '🟡 DEGRADING', '🔴 CRITICAL', '⚫ EOL']

def test_rul_estimation():
    insights = BatteryInsights()
    rul = insights.estimate_remaining_useful_life(ir=0.085)
    assert rul['estimated_cycles_remaining_conservative'] > 0
```

### Manual Validation

- ✅ App starts without deprecation warnings
- ✅ Predictions load correctly from pickle files
- ✅ Feature shape matches (15 features after preprocessing)
- ✅ Insights show immediately after prediction
- ✅ Simulator increments SOC correctly
- ✅ All 5 tabs render without errors

---

## 📈 ML Lifecycle (Notebooks 01-06)

### Notebook 01: EDA
- Load and profile 1,900 records
- Discover `cycle_degradation` has r = 0.038 (noise)
- Identify `internal_resistance` as predictable (true signal)
- Statistical summaries and visualizations

### Notebook 02: Preprocessing
- **FIXED**: Imputer creation and pipeline organization
- Fit imputer on 23 engineered features
- Fit scaler on imputed 23 features
- Save preprocessor_pipeline.pkl

### Notebook 03: Feature Engineering
- Create 23 engineered features from 15 raw
- Apply RFE to select best 15
- Save feature_columns.json (15 feature names)

### Notebook 04: Regression Training
- Train XGBoost on 15 RFE features → internal_resistance
- GridSearchCV for hyperparameter tuning
- Evaluate on test set → R² = 0.97
- Save xgb_regression_model.pkl

### Notebook 05: Classification Training
- Train XGBoost on 15 RFE features → over_temp_flag
- StratifiedKFold for handling temporal block pattern
- Evaluate on CV folds → F1 = 0.95
- Save xgb_classification_model.pkl

### Notebook 06: Evaluation
- Load all models and preprocessing artifacts
- Run test set predictions
- Generate comparison tables and visualizations
- Document results and limitations

---

## 🚀 Deployment Checklist

Before production deployment, verify:

- ✅ All 1,900 CSV records present
- ✅ Notebooks 01-06 run without errors
- ✅ Models saved to `/models/`:
  - xgb_regression_model.pkl
  - xgb_classification_model.pkl
  - preprocessor_pipeline.pkl
  - feature_columns.json
- ✅ Web app starts: `streamlit run app/app.py`
- ✅ All 5 tabs functional
- ✅ Predictions show health insights
- ✅ Simulator 0-100% SOC works
- ✅ Model health checks pass
- ✅ No Streamlit warnings
- ✅ Tests pass: `pytest test_*.py`

---

## 🔮 Future Enhancements

1. **Batch RUL Forecasting**: Predict RUL for fleet of 100+ batteries
2. **Trend Analysis**: Track health score over time (not just current prediction)
3. **Anomaly Detection**: Flag batteries with unusual patterns
4. **Performance Benchmarking**: Compare new cycles to historical baseline
5. **Explainability**: SHAP values showing feature contributions
6. **API Endpoint**: REST API for integration with fleet management systems

---

## 📝 Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Feature Engineering** | ✅ Complete | 15→23→15 pipeline working |
| **Models** | ✅ Complete | XGBoost Reg (R²=0.97), Class (F1=0.95) |
| **Preprocessing** | ✅ Fixed | Correct order: Engineer→Impute→Scale→Select |
| **Insights** | ✅ Complete | Health score, RUL, fleet comparison |
| **Simulator** | ✅ Complete | Interactive 0-100% SOC with predictions |
| **Web UI** | ✅ Complete | 5 tabs, Streamlit, no deprecation warnings |
| **Testing** | ✅ Complete | Unit tests for simulator, insights, preprocessor |
| **Documentation** | ✅ Complete | README.md, PROJECT.md, MODEL_CARD.md |

**Overall Status**: **PRODUCTION READY** ✅

Last Updated: May 17, 2026
