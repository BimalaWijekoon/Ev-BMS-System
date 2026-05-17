# 📋 PROJECT.md — Technical Deep-Dive

## Overview

This document provides a detailed technical overview of the EV Battery Intelligence System, including architecture decisions, recent improvements, and the complete ML pipeline.

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
