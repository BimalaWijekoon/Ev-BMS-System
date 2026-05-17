---
title: EV Battery Intelligence Predictor
emoji: 🔋
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32
python_version: 3.12
app_file: app.py
pinned: false
---

# 🔋 EV Battery Intelligence System

**Advanced ML-powered battery diagnostics for electric vehicles** — Real-time internal resistance prediction, health scoring, RUL estimation, and fleet analytics.

---

## 📊 Executive Summary

This system predicts EV battery degradation **in real-time** using machine learning models trained on 1,900 charging cycles. It combines:

- ✅ **XGBoost Regression** for internal resistance prediction (R² = 0.97)
- ✅ **XGBoost Classification** for temperature safety flags (F1 = 0.95)
- ✅ **Battery Health Scoring** (0-100%) based on component degradation
- ✅ **RUL Estimation** — remaining useful cycles until replacement threshold
- ✅ **Fleet Analytics** — percentile ranking and degradation trends
- ✅ **Interactive Simulation** — play battery charge cycle 0→100% with live predictions

**Use Case**: Maintenance teams can optimize battery replacement schedules, reduce warranty costs, and prevent thermal runaway incidents.

---

## 🎯 What's New in This Release

### ✨ Major Features (Completed)

| Feature | Impact | Status |
|---------|--------|--------|
| **Battery Health Insights** | Users see instant health score + RUL after predictions | ✅ Production |
| **Interactive Simulator** | Simulate 0-100% SOC charging with real-time ML predictions | ✅ Production |
| **Fleet Comparison** | Benchmark battery vs 1900-unit fleet average | ✅ Production |
| **Model Health Checks** | Diagnostic endpoint to verify all components load correctly | ✅ Production |
| **Streamlit Web UI** | 5-tab interactive interface (Gradio → Streamlit migration) | ✅ Production |

### 🔧 Recent Fixes & Improvements

| Issue | Root Cause | Solution | Impact |
|-------|-----------|----------|--------|
| **Feature Shape Mismatch** | Pipeline selected 15 features BEFORE scaling/imputing | Reordered: Engineer (23) → Impute (23) → Scale (23) → Select (15) | Models work correctly |
| **Imputer None Error** | Notebook had comment but never created imputer object | Implemented proper imputer creation in Notebook 02 Cell 8 | Preprocessing runs without errors |
| **Streamlit Deprecation** | `use_container_width=True` removed in Streamlit v1.x | Replaced with `width='stretch'` and `width='content'` | Zero warnings on startup |
| **Insights UX** | Separate tab forced context switching | Integrated insights into Tab 1 (Single Prediction) | Better user workflow |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB INTERFACE                      │
│  5 Tabs: Prediction | Batch | Info | Health | Simulation       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐  ┌──────────┐  ┌──────────────┐
   │ Predict │  │Simulate  │  │  Insights    │
   │ Module  │  │  Module  │  │   Engine     │
   └────┬────┘  └────┬─────┘  └──────┬───────┘
        │             │              │
        └─────────────┼──────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
   ┌──────────────────┐      ┌──────────────────┐
   │ Preprocessing    │      │  2 XGBoost       │
   │ Pipeline         │      │  Models          │
   │ (23→15 features) │      │  (Reg + Class)   │
   └──────────────────┘      └──────────────────┘
```

---

## 📁 Project Structure

```
ev-battery-ml/
│
├── 📓 NOTEBOOKS (Full ML Lifecycle)
│   ├── 01_data_understanding_eda.ipynb         ← Data profiling
│   ├── 02_data_preprocessing.ipynb             ← Imputer + Scaler
│   ├── 03_feature_engineering.ipynb            ← 23→15 features via RFE
│   ├── 04_model_training_regression.ipynb      ← XGBRegressor training
│   ├── 05_model_training_classification.ipynb  ← XGBClassifier training
│   └── 06_model_evaluation_and_selection.ipynb ← Final test evaluation
│
├── 🔧 SRC (Production Code)
│   ├── preprocessor.py              ← BatteryPreprocessor class
│   ├── feature_engineer.py          ← Feature engineering
│   ├── train_regression.py          ← Training scripts
│   ├── train_classification.py      ← Training scripts
│   ├── predict.py                   ← Inference & health checks
│   ├── simulator.py                 ← BatterySimulator (0-100% SOC)
│   └── insights.py                  ← Health scoring, RUL, fleet comparison
│
├── 🌐 WEB APP
│   └── app.py                       ← Streamlit: 5-tab interface
│
├── 📦 MODEL ARTIFACTS
│   ├── xgb_regression_model.pkl
│   ├── xgb_classification_model.pkl
│   ├── preprocessor_pipeline.pkl    ← Scaler + Imputer
│   ├── feature_columns.json         ← 15 RFE-selected names
│   └── [metrics, configs, model card]
│
├── 📊 DATA
│   └── nev_battery_charging.csv     ← 1,900 cycles
│
├── 🧪 TESTS
│   ├── test_simulator.py
│   ├── test_insights.py
│   └── test_preprocessor.py
│
└── 📄 DOCS
    ├── README.md
    ├── PROJECT.md
    ├── INSIGHTS_ARCHITECTURE.md
    └── models/MODEL_CARD.md
```

---

## 🚀 Features & Tabs

### ⚡ Tab 1: Single Prediction
**Workflow**: Enter 15 battery features → Run → Get predictions + **insights**

**Displays**:
- IR prediction gauge
- Over-temp probability
- **→ Integrated insights**:
  - Health score (0-100%)
  - Component breakdown
  - Active alerts
  - RUL estimate
  - Fleet comparison
  - Recommendations

### 📊 Tab 2: Batch Analysis
**Upload CSV** → Multi-battery predictions + distribution charts

### 📖 Tab 3: Model Info
**Transparency**: Architecture, features, metrics, dataset details

### 🏥 Tab 4: Model Health
**Diagnostics**: Verify all components (models, preprocessor) load correctly

### 🔋 Tab 5: Charging Simulation
**Interactive**: Simulate 0→100% SOC with live IR & safety predictions

---

## 🧠 How It Works

### Data Pipeline
```
Raw Input (15 features)
    ↓ Engineer (23 features)
    ↓ Impute (median strategy)
    ↓ Scale (StandardScaler)
    ↓ Select RFE (best 15 from 23)
    ↓ XGBoost Inference
    ↓ Health Scoring + RUL + Insights
    ↓ Web Display
```

### Health Score Formula
```
Score = 40% × IR_health + 35% × Thermal_health + 25% × Voltage_health

Status:
  80-100: 🟢 HEALTHY
  60-80:  🟡 DEGRADING
  40-60:  🔴 CRITICAL
  0-40:   ⚫ EOL
```

### Models
- **Regression**: XGBoost (R² = 0.97) → Internal Resistance
- **Classification**: XGBoost (F1 = 0.95) → Over-Temperature Flag
- **Over-Voltage**: Rule-based (4.15V threshold)

---

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 2. Install
pip install -r requirements.txt

# 3. Run (optional - notebooks)
jupyter notebook notebooks/  # 01 → 06

# 4. Launch app
streamlit run app/app.py
```

## 🛠️ Technologies

Python 3.10+ | Streamlit | XGBoost | scikit-learn | Plotly | Pandas

## 📄 License

MIT License
