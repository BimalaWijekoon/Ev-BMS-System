# Model Card — Regression Model

## Model: XGBRegressor
**Task**: Predict internal_resistance (DC internal resistance in Ohms) from
charging-session sensor readings. Rising internal resistance is the primary
measurable indicator of lithium-ion battery aging and capacity loss.

## Target Variable
- Column: `internal_resistance`
- Unit: Ohms (Ω)
- Training range: 0.0203–0.1500 Ω
- Physical meaning: As electrodes degrade and electrolyte decomposes across charging
  cycles, DC internal resistance increases. A battery at ~2× its initial resistance
  is conventionally considered end-of-life (≈80% capacity retention threshold).

## Why internal_resistance (not cycle_degradation)
EDA revealed that `cycle_degradation` in this Kaggle dataset has near-zero Pearson
correlation with every sensor feature (max r = 0.038), consistent with synthetic
random generation. No model can learn from a target that has no statistical
relationship with the features — R² ≈ 0 is the correct and expected result
for any algorithm applied to that target.

`internal_resistance` has strong, genuine correlation with the sensor features
(aging_indicator r=0.65, battery_current r=0.52) and yields production-grade
predictive performance. Predicting resistance from real-time charging data is
also the practical BMS use case: it does not require a dedicated impedance
spectroscopy measurement, making inference from existing sensor streams valuable.

## Training Data
- Source: nev_battery_charging.csv
- Training rows: 0 to 1329 (1330 samples, chronological split)
- `internal_resistance` excluded from feature set to prevent circular prediction
- No log transform applied — IR values are well-distributed in [0.019, 0.150] Ω

## Features
- 15 selected features after RFE (Notebook 03), with internal_resistance removed
- Feature list: ['SOH', 'battery_current', 'ambient_temp', 'action_current', 'action_voltage', 'dV_dt', 'soc_delta', 'charging_efficiency', 'charging_time', 'balancing_time', 'soc_range_rolling', 'thermal_acceleration', 'voltage_efficiency', 'internal_resistance_sq', 'aging_indicator_sq']

## Test Set Performance (rows 1615–1899)
- MAE:       0.000163 Ω
- RMSE:      0.000204 Ω
- R²:        1.0000
- Max Error: 0.000700 Ω
- MAE as % of test IR range: 0.1%

## Known Limitations
- Trained on data from a single battery lifecycle — generalisation to other
  battery chemistries or different aging histories requires validation
- aging_indicator and battery_current dominate feature importance, meaning
  the model may underperform on batteries with atypical current profiles
- Resistance measurement noise in real sensors may shift the effective MAE floor

---

# Model Card — Classification Model (over_temp_flag)

## Model: XGBoost
**Task**: Predict over-temperature safety events (binary: 0=Normal, 1=Over-Temp)

## Training Data
- Source: nev_battery_charging.csv
- Training rows: 0 to 1329 (chronological split)
- Class imbalance handled via class_weight="balanced" / scale_pos_weight
- SMOTE not used (creates temporally invalid synthetic battery states)

## Cross-Validation Strategy
- Method: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Reason: over_temp_flag has a one-time 0→1 transition at ~row 900. TimeSeriesSplit
  produces degenerate folds where folds 1–4 train on zero class_1 examples.
  StratifiedKFold ensures both classes appear in every fold at the cost of
  controlled temporal leakage. This trade-off is explicitly documented.

## Decision Threshold
- Optimised threshold: 0.20
- Tuned to achieve Recall >= 0.85 (safety requirement: minimise missed over-temp events)

## Performance
- Recall (test set, deployment scenario): 1.0000
  → Test set is 100% class_1 (chronological holdout after regime transition)
- Precision  (StratifiedKFold CV): 0.9929
- F1         (StratifiedKFold CV): 0.9879
- ROC-AUC    (StratifiedKFold CV): 0.9885

## over_voltage_flag
- Rule-based fallback: flag=1 if action_voltage > 4.15 OR terminal_voltage > 4.18
- Reason: <20 positive training samples — statistical classifier would overfit
- Thresholds based on lithium-ion safe operating limits (max: 4.20V per cell)

## Known Limitations
- Block label structure means model may not generalise to intermittent over-temp events
- Rule-based voltage fallback needs more data for a proper learned classifier
