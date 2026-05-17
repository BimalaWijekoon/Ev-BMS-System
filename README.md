# Ev-BMS-System

🔋 **EV Battery Management System** — A comprehensive machine learning pipeline for predicting battery internal resistance (aging proxy) and safety flags from EV charging sensor data.

## Features
- 6 Jupyter notebooks covering the full ML lifecycle (EDA → Preprocessing → Feature Engineering → Training → Evaluation)
- Regression model for internal resistance prediction (R² ≈ 0.97)
- Classification models for over-temperature detection (StratifiedKFold CV)
- Rule-based over-voltage detection (domain-knowledge threshold)
- Gradio web application for real-time predictions
- Domain-knowledge-driven feature engineering

## Quick Start
```bash
cd ev-battery-ml
python -m venv venv
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
jupyter notebook notebooks/  # Run notebooks 01-06 sequentially
python app/app.py            # Launch Gradio app
```

See [ev-battery-ml/README.md](ev-battery-ml/README.md) for full documentation.
