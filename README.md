

<div align="center">

# 🔋 EV Battery Intelligence System

**Production-grade ML system for real-time EV battery health diagnostics**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HF%20Spaces-FF4B4B?style=for-the-badge&logo=streamlit)](https://bmwmiuranda-ev-battery-intelligence.hf.space)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Space-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/bmwmiuranda/ev-battery-intelligence)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/BimalaWijekoon/Ev-BMS-System)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-E26E1B?style=flat-square)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

<img width="1905" height="851" alt="image" src="https://github.com/user-attachments/assets/acd4e233-7bef-44d5-b683-1680b4f5b3bb" />


The **EV Battery Intelligence System** is a complete end-to-end machine learning pipeline that predicts EV battery degradation in real time. Built with **XGBoost** models trained on **1,900 charging cycles**, it enables maintenance teams to:

- Predict **internal resistance** (the primary electrochemical aging signal) with **R² = 0.9999**
- Detect **thermal overtemperature events** with **F1 = 0.988** before they cause damage
- Estimate **Remaining Useful Life (RUL)** and compare batteries against fleet benchmarks
- Simulate an **interactive 0→100% charging cycle** with live ML predictions at every step

> **Use case**: Fleet operators, BMS engineers, and EV manufacturers who need predictive maintenance insights to reduce warranty costs and prevent thermal runaway incidents.

---

## 🎯 Model Performance

### Regression — Internal Resistance Prediction

| Model | R² Score | MAE (Ω) | RMSE (Ω) |
|-------|----------|----------|----------|
| Linear Regression | 0.9641 | 0.006299 | 0.007432 |
| Random Forest | 0.9985 | 0.001043 | 0.001516 |
| **XGBoost ✅ Selected** | **0.9999** | **0.000169** | **0.000221** |

### Classification — Over-Temperature Flag Detection

| Model | F1 Score | Recall | Precision | ROC-AUC |
|-------|----------|--------|-----------|---------|
| Logistic Regression | 0.9863 | 0.9782 | 0.9945 | 0.9908 |
| SVM-RBF | 0.9827 | 0.9938 | 0.9719 | 0.9398 |
| **XGBoost ✅ Selected** | **0.9879** | **0.9829** | **0.9929** | **0.9885** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT WEB INTERFACE (5 Tabs)                  │
│   Single Prediction │ Batch Analysis │ Model Info │ Health │ Sim    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐    ┌──────────┐    ┌─────────────┐
        │ predict  │    │simulator │    │  insights   │
        │   .py    │    │   .py    │    │    .py      │
        └────┬─────┘    └────┬─────┘    └──────┬──────┘
             │               │                 │
             └───────────────┼─────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌──────────────────────┐    ┌────────────────────────┐
    │  BatteryPreprocessor │    │   XGBoost Models       │
    │  17 raw → engineer   │    │   ① Regression (IR)    │
    │  23 feats → impute   │    │   ② Classification     │
    │  → scale → RFE(15)   │    │      (Over-Temp)       │
    └──────────────────────┘    └────────────────────────┘
```

---

## 🚀 Live Demo

**Try it now** → [https://bmwmiuranda-ev-battery-intelligence.hf.space](https://bmwmiuranda-ev-battery-intelligence.hf.space)

### Tab 1 — ⚡ Single Prediction
Enter 11 real-time sensor readings → get instant predictions:
- **Internal Resistance** (mΩ) via gauge
- **Over-Temperature Risk** (%) via gauge
- **Battery Health Score** (0–100%)
- **RUL estimate** (remaining charge cycles)
- **Fleet comparison** (percentile ranking vs 1,900-unit fleet)
- **Actionable recommendations**

### Tab 2 — 📊 Batch Analysis
Upload a CSV of battery sensor logs → bulk predictions with distribution charts + downloadable results CSV.

### Tab 3 — 📖 Model Info
Full transparency: model architecture, feature selection rationale, R²/F1 metrics, dataset statistics.

### Tab 4 — 🏥 Model Health Check
Live diagnostic of all components: preprocessor pipeline, regression model, classification model, and config files.

### Tab 5 — 🔋 Charging Simulation
Step through a complete 0→100% charge cycle with real-time IR and safety predictions at every SOC increment. Watch the battery degrade in real time.

---

## 🧠 ML Pipeline

```
Raw Sensor Input (17 features)
    │
    ▼ feature_engineer.py
    Engineer 23 features:
    ├── delta_internal_resistance   ← IR growth from training baseline
    ├── soc_range_rolling           ← Cycling aggressiveness (window=10)
    ├── thermal_acceleration        ← dT/dt × battery_temp compound feature
    ├── voltage_efficiency          ← action_voltage / terminal_voltage
    └── polynomial terms (IR², aging²)
    │
    ▼ SimpleImputer (median strategy)
    Fill any missing values using training medians
    │
    ▼ StandardScaler
    Normalize all 23 engineered features
    │
    ▼ RFE Selection (15 best from 23)
    SOH, battery_current, ambient_temp, action_current, action_voltage,
    dV_dt, soc_delta, charging_efficiency, charging_time, balancing_time,
    soc_range_rolling, thermal_acceleration, voltage_efficiency,
    internal_resistance_sq, aging_indicator_sq
    │
    ├──▶ XGBRegressor  →  internal_resistance (Ω)
    └──▶ XGBClassifier →  over_temp_flag (0/1)  +  over_voltage (rule-based)
```

### Health Score Formula
```
Health Score = 40% × IR_health + 35% × Thermal_health + 25% × Voltage_health

Thresholds:
  80–100  🟢 HEALTHY   — Continue normal operation
  60–80   🟡 DEGRADING — Increase monitoring frequency
  40–60   🔴 CRITICAL  — Daily monitoring + reduce charging power
  0–40    ⚫ EOL       — Replace immediately
```

---

## 📁 Project Structure

```
Ev-BMS-System/
└── ev-battery-ml/
    ├── 📓 notebooks/                        ← Full ML lifecycle (6 notebooks)
    │   ├── 01_data_understanding_eda.ipynb  ← EDA & profiling
    │   ├── 02_data_preprocessing.ipynb      ← Imputer + Scaler pipeline
    │   ├── 03_feature_engineering.ipynb     ← 23→15 features via RFE
    │   ├── 04_model_training_regression.ipynb    ← XGBRegressor + GridSearchCV
    │   ├── 05_model_training_classification.ipynb ← XGBClassifier + StratifiedKFold
    │   └── 06_model_evaluation_and_selection.ipynb ← Final evaluation
    │
    ├── 🔧 src/                              ← Production inference code
    │   ├── preprocessor.py                  ← BatteryPreprocessor class
    │   ├── feature_engineer.py              ← Feature engineering functions
    │   ├── predict.py                       ← Inference + model health checks
    │   ├── simulator.py                     ← BatterySimulator (0→100% SOC)
    │   ├── insights.py                      ← Health scoring, RUL, fleet stats
    │   ├── train_regression.py              ← Standalone training script
    │   └── train_classification.py          ← Standalone training script
    │
    ├── 🌐 app/
    │   └── app.py                           ← Streamlit 5-tab web interface
    │
    ├── 📦 models/                           ← Saved artifacts
    │   ├── regression_model.pkl             ← XGBRegressor (~255 KB)
    │   ├── classification_model_temp.pkl    ← XGBClassifier (~383 KB)
    │   ├── preprocessor_pipeline.pkl        ← Scaler + Imputer (~3 KB)
    │   ├── feature_columns.json             ← 15 RFE-selected feature names
    │   ├── classification_config.json       ← Threshold + class weights
    │   ├── regression_results.json          ← Model comparison metrics
    │   ├── classification_results.json      ← Model comparison metrics
    │   └── MODEL_CARD.md                    ← Model documentation
    │
    ├── 📊 data/
    │   └── nev_battery_charging.csv         ← 1,900 charging cycles dataset
    │
    ├── 📈 plots/                            ← Training & evaluation visualizations
    │
    ├── requirements.txt                     ← Runtime dependencies
    └── README.md                            ← This file
```

---

## ⚡ Quick Start

### Option 1 — Try it online (no install needed)
👉 **[https://bmwmiuranda-ev-battery-intelligence.hf.space](https://bmwmiuranda-ev-battery-intelligence.hf.space)**

### Option 2 — Run locally

```bash
# Clone the repo
git clone https://github.com/BimalaWijekoon/Ev-BMS-System.git
cd Ev-BMS-System/ev-battery-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# .\venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

### Option 3 — Retrain from scratch

```bash
# Run notebooks in order
jupyter notebook notebooks/
# Execute: 01 → 02 → 03 → 04 → 05 → 06
# Model artifacts saved to models/
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Web App** | Streamlit 1.44, Plotly 5.x |
| **ML Models** | XGBoost 2.0, scikit-learn 1.5 |
| **Data Processing** | Pandas 2.1, NumPy 1.26 |
| **Feature Engineering** | RFE (Recursive Feature Elimination) |
| **Preprocessing** | StandardScaler + SimpleImputer (median) |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |
| **Deployment** | Hugging Face Spaces (CPU Basic, Python 3.13) |
| **Language** | Python 3.13 |

---

## 📊 Dataset

- **Source**: NEV Battery Charging Dataset
- **Size**: 1,900 charging cycle observations
- **Features**: 17 raw sensor measurements per cycle
- **Targets**:
  - `internal_resistance` (Ω) — regression target
  - `over_temp_flag` (0/1) — classification target
  - `over_voltage_flag` (0/1) — rule-based (> 4.15V threshold)
- **Split**: TimeSeriesSplit (chronological) — no data leakage

---

## 🗺️ Roadmap

- [x] EDA & data profiling
- [x] Feature engineering pipeline (23 features)
- [x] RFE feature selection (15 best)
- [x] XGBoost regression (R² = 0.9999)
- [x] XGBoost classification (F1 = 0.988)
- [x] Streamlit 5-tab web interface
- [x] Battery health scoring & RUL estimation
- [x] Fleet comparison analytics
- [x] Interactive charging simulation
- [x] Production deployment on Hugging Face Spaces
- [ ] REST API endpoint (FastAPI)
- [ ] Real-time streaming data ingestion
- [ ] Multi-cell pack-level analysis

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.


---

<div align="center">

**⭐ If this project helped you, please give it a star on GitHub!**

[![GitHub stars](https://img.shields.io/github/stars/BimalaWijekoon/Ev-BMS-System?style=social)](https://github.com/BimalaWijekoon/Ev-BMS-System/stargazers)

</div>
