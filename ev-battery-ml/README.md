---
title: EV Battery Degradation Predictor
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: app/app.py
pinned: false
---

# 🔋 EV Battery Management System — ML Pipeline

A comprehensive machine learning pipeline for predicting EV battery degradation and safety flags from charging sensor data.

## 📊 Project Overview

This project implements a complete ML lifecycle for battery management:

| Component | Description |
|-----------|-------------|
| **Dataset** | 1,900 records × 21 features from EV battery charging cycles |
| **Regression Target** | `cycle_degradation` — battery degradation per charge cycle |
| **Classification Targets** | `over_temp_flag` and `over_voltage_flag` — safety alerts |
| **Models** | Linear Regression, Random Forest, XGBoost, Logistic Regression, SVM-RBF |
| **App** | Gradio web interface for real-time predictions |

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
- Target variable distribution analysis
- Temporal pattern discovery (longitudinal battery lifecycle)
- Correlation heatmap and high-correlation pair identification

### 2. Preprocessing
- **Chronological split** (70/15/15) — prevents temporal data leakage
- StandardScaler fit on training data only
- Log1p transform for regression target
- Class imbalance handling via balanced weights (not SMOTE — preserves temporal order)

### 3. Feature Engineering
- 5 domain-knowledge features: delta IR, SOC range, thermal acceleration, voltage efficiency, polynomial terms
- Filter-based selection (correlation threshold)
- RFE with RandomForest (wrapper method)

### 4. Model Training
- **Regression**: Linear → Random Forest → XGBoost (with GridSearchCV + TimeSeriesSplit)
- **Classification**: Logistic Regression → SVM-RBF → XGBoost (with threshold tuning)
- Early stopping and learning curve analysis

### 5. Evaluation
- Final test set evaluation with full visualization suite
- Model comparison tables
- Professional model cards

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

## 📈 Key Findings

- **battery_temp** is a cumulative thermal proxy (30→1129), NOT session temperature
- **over_temp_flag** has a temporal block pattern — class split at ~row 900
- **over_voltage_flag** has extreme imbalance — rule-based fallback used
- **TimeSeriesSplit** used throughout — no random shuffling allowed

## 🛠️ Technologies

- Python 3.10+ | pandas | NumPy | scikit-learn | XGBoost
- Matplotlib | Seaborn | Gradio | Jupyter

## 📄 License

MIT License
