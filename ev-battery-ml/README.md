---
title: EV Battery Intelligence Predictor
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: app/app.py
pinned: false
---

# 🔋 EV Battery Management System — ML Pipeline

A comprehensive machine learning pipeline for predicting EV battery internal resistance (electrochemical aging proxy) and safety flags from charging sensor data.

## 📊 Project Overview

This project implements a complete ML lifecycle for battery management:

| Component | Description |
|-----------|-------------|
| **Dataset** | 1,900 records × 21 features from EV battery charging cycles |
| **Regression Target** | `internal_resistance` — primary electrochemical aging indicator (Ω) |
| **Classification Targets** | `over_temp_flag` (ML classifier) and `over_voltage_flag` (rule-based) |
| **Models** | Linear Regression, Random Forest, XGBoost, Logistic Regression, SVM-RBF |
| **App** | Gradio web interface for real-time predictions |

### Why Internal Resistance, Not Cycle Degradation?

EDA re-analysis revealed that `cycle_degradation` has near-zero correlation with all features (max Pearson r = 0.038), consistent with synthetic noise in the Kaggle dataset. `internal_resistance` achieves **R² = 0.97** with Random Forest on the held-out test set — a production-grade result. Internal resistance is the primary measurable indicator of lithium-ion battery aging: as SEI layer grows and electrolyte degrades, impedance rises monotonically. This makes it both physically meaningful and statistically predictable.

## 🏗️ Project Structure

```
ev-battery-ml/
├── data/                           # Dataset
│   └── nev_battery_charging.csv
├── notebooks/                      # 6 Jupyter notebooks (full ML lifecycle)
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_regression.ipynb
│   ├── 05_model_training_classification.ipynb
│   └── 06_model_evaluation_and_selection.ipynb
├── src/                            # Reusable Python modules
│   ├── preprocessor.py             # BatteryPreprocessor class
│   ├── feature_engineer.py         # Feature engineering functions
│   ├── train_regression.py         # Regression training script
│   ├── train_classification.py     # Classification training script
│   └── predict.py                  # Inference module
├── models/                         # Saved model artifacts
│   ├── regression_model.pkl
│   ├── classification_model_temp.pkl
│   ├── preprocessor_pipeline.pkl
│   ├── feature_columns.json
│   └── MODEL_CARD.md
├── plots/                          # All generated visualizations
├── app/
│   └── app.py                      # Gradio web interface
└── requirements.txt
```

## 🔬 ML Pipeline

### 1. Data Understanding (EDA)
- Statistical profiling with tail percentiles
- Target variable distribution and correlation analysis
- Temporal pattern discovery (longitudinal battery lifecycle)
- Identification of `cycle_degradation` as unpredictable synthetic noise (r = 0.038)

### 2. Preprocessing
- **Chronological split** (70/15/15) — prevents temporal data leakage
- StandardScaler fit on training data only
- No log transform needed — IR values (0.019–0.150 Ω) are already clean
- `internal_resistance` excluded from features to prevent target leakage

### 3. Feature Engineering
- 5 domain-knowledge features: delta IR, SOC range, thermal acceleration, voltage efficiency, polynomial terms
- Filter-based selection (correlation threshold)
- RFE with RandomForest (wrapper method)

### 4. Model Training
- **Regression** (internal_resistance): Linear → Random Forest → XGBoost (GridSearchCV + TimeSeriesSplit)
- **Classification** (over_temp_flag): Logistic Regression → SVM-RBF → XGBoost (StratifiedKFold CV)
- Classification uses StratifiedKFold because `over_temp_flag` has a temporal block pattern (0→1 at row ~900), making chronological val/test sets single-class. This is explicitly documented as an established practice for block-structured temporal labels.
- `over_voltage_flag`: Rule-based fallback (action_voltage > 4.15 OR terminal_voltage > 4.18) due to <20 positive training examples

### 5. Evaluation
- Final test set evaluation with full visualization suite
- Model comparison tables
- Professional model cards with limitations documented

## 📈 Key Findings

- **internal_resistance** is the true aging signal — R² = 0.97 with RF, 0.84 with Linear Regression
- **cycle_degradation** is unpredictable noise (max r = 0.038 with any feature)
- **battery_temp** is a cumulative thermal proxy (30→1129), NOT session temperature
- **over_temp_flag** has a temporal block pattern — class transition at ~row 900
- **over_voltage_flag** has extreme imbalance — rule-based fallback used
- StratifiedKFold used for classification CV (documented deviation from chronological split)

## 🚀 Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run notebooks sequentially (01 → 06)
jupyter notebook notebooks/

# Launch Gradio app (after running notebooks)
python app/app.py
```

## 🛠️ Technologies

- Python 3.10+ | pandas | NumPy | scikit-learn | XGBoost
- Matplotlib | Seaborn | Gradio | Jupyter

## 📄 License

MIT License
