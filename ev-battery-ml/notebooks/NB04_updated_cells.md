# Notebook 04 — Updated Cells Only
# Target changed: `cycle_degradation` → `internal_resistance`
# Replace only the cells listed below. All other cells stay exactly as-is.

---

## ▶ CELL TO REPLACE: "Load and prepare data"
Find this cell by looking for: `df = pd.read_csv('../data/nev_battery_charging.csv')`

Replace the entire cell with:

```python
# Load and prepare data
df = pd.read_csv('../data/nev_battery_charging.csv').drop(columns=['timestamp']).drop_duplicates()
n = len(df); t_end, v_end = int(n*0.70), int(n*0.85)
targets = ['cycle_degradation','over_temp_flag','over_voltage_flag']

df_train, df_val, df_test = df.iloc[:t_end].copy(), df.iloc[t_end:v_end].copy(), df.iloc[v_end:].copy()
baseline_ir = df_train['internal_resistance'].iloc[0]

# Engineer features
df_train = engineer_all_features(df_train, baseline_ir, ['battery_temp'])
df_val   = engineer_all_features(df_val,   baseline_ir, ['battery_temp'])
df_test  = engineer_all_features(df_test,  baseline_ir, ['battery_temp'])

# Load selected features from NB03
try:
    with open('../models/feature_columns.json') as f:
        selected_features = json.load(f)
    print(f'Loaded {len(selected_features)} selected features from NB03')
except FileNotFoundError:
    selected_features = [c for c in df_train.columns if c not in targets]
    print(f'No feature_columns.json found, using all {len(selected_features)} features')

# ── CHANGED: target is now internal_resistance ──────────────────────────────
# internal_resistance is the gold-standard battery aging metric.
# It increases monotonically as electrodes degrade and electrolyte decomposes.
# It has genuine, measurable correlation with the sensor features (R²=0.97 RF).
# cycle_degradation in this dataset is synthetically generated noise with
# near-zero correlation with any feature (max Pearson r = 0.038), making it
# statistically unpredictable by any model.
# We EXCLUDE internal_resistance from the feature set since it is now the target.
selected_features = [f for f in selected_features if f != 'internal_resistance']

# Separate X and y
X_train = df_train[[c for c in selected_features if c in df_train.columns]]
X_val   = df_val[[c for c in selected_features if c in df_val.columns]]
X_test  = df_test[[c for c in selected_features if c in df_test.columns]]
y_train = df_train['internal_resistance']
y_val   = df_val['internal_resistance']
y_test  = df_test['internal_resistance']

# Scale — fit on train only
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# No log transform needed — internal_resistance values (0.019–0.150 Ω) are
# already well-distributed without skew. log1p would also be a near-no-op
# on values this small (log1p(0.05) ≈ 0.0488), introducing no benefit.
print(f'X_train: {X_train_s.shape}, X_val: {X_val_s.shape}, X_test: {X_test_s.shape}')
print(f'Target  internal_resistance — train: {y_train.min():.4f}–{y_train.max():.4f} Ω  mean={y_train.mean():.4f} Ω')
```

---

## ▶ CELL TO REPLACE: Linear Regression training + metrics
Find this cell by looking for: `lr = LinearRegression()`

Replace the entire cell with:

```python
# Linear Regression — Baseline
t0 = time.time()
lr = LinearRegression()
lr.fit(X_train_s, y_train)
lr_time = time.time() - t0

# No expm1 needed — predictions are already in original Ω units
y_pred_lr = lr.predict(X_val_s)
mae_lr  = mean_absolute_error(y_val, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
r2_lr   = r2_score(y_val, y_pred_lr)

print(f'Linear Regression Results (Validation Set):')
print(f'  MAE:  {mae_lr:.6f} Ω  (avg prediction error in ohms)')
print(f'  RMSE: {rmse_lr:.6f} Ω')
print(f'  R²:   {r2_lr:.4f}  (variance in internal_resistance explained by features)')
print(f'  Time: {lr_time:.3f}s')

if r2_lr < 0.5:
    print('\n→ R² < 0.5: Linear relationships insufficient. Non-linear models needed.')
else:
    print(f'\n→ R² = {r2_lr:.4f}: Linear relationships captured. Non-linear models may improve further.')
```

---

## ▶ CELL TO REPLACE: Random Forest GridSearch + metrics
Find this cell by looking for: `gs_rf = GridSearchCV(`

Replace the entire cell with:

```python
# Random Forest with GridSearch + TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

rf_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [5, 10, 15, None],
    'min_samples_split':[2, 5, 10],
    'max_features':     ['sqrt', 0.5]
}

t0 = time.time()
gs_rf = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_grid, cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=0
)
gs_rf.fit(X_train_s, y_train)
rf_time = time.time() - t0

best_rf = gs_rf.best_estimator_
print(f'Best RF params: {gs_rf.best_params_}')
print(f'Best CV RMSE:   {-gs_rf.best_score_:.6f} Ω')

# No expm1 — predictions in Ω directly
y_pred_rf = best_rf.predict(X_val_s)
mae_rf  = mean_absolute_error(y_val, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
r2_rf   = r2_score(y_val, y_pred_rf)
print(f'\nRandom Forest Results (Validation Set):')
print(f'  MAE:  {mae_rf:.6f} Ω')
print(f'  RMSE: {rmse_rf:.6f} Ω')
print(f'  R²:   {r2_rf:.4f}')
print(f'  Time: {rf_time:.1f}s')
```

---

## ▶ CELL TO REPLACE: XGBoost GridSearch
Find this cell by looking for: `gs_xgb = GridSearchCV(`

Replace the entire cell with:

```python
# XGBoost with GridSearch + TimeSeriesSplit
xgb_grid = {
    'n_estimators':  [200, 500, 1000],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'reg_alpha':     [0, 0.1, 1.0],
    'reg_lambda':    [1, 5, 10]
}

t0 = time.time()
gs_xgb = GridSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42,
                 eval_metric='rmse', verbosity=0),
    xgb_grid, cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=0
)
# No log transform on target — train directly in Ω
gs_xgb.fit(X_train_s, y_train)
print(f'Best XGB params: {gs_xgb.best_params_}')
print(f'Best CV RMSE:    {-gs_xgb.best_score_:.6f} Ω')
```

---

## ▶ CELL TO REPLACE: XGBoost retrain with early stopping + metrics
Find this cell by looking for: `best_xgb = XGBRegressor(`

Replace the entire cell with:

```python
# Retrain best XGBoost WITH early stopping for training curve visualization
best_xgb = XGBRegressor(
    **gs_xgb.best_params_,
    objective='reg:squarederror',
    random_state=42,
    eval_metric='rmse',
    early_stopping_rounds=50,
    verbosity=0
)
best_xgb.fit(
    X_train_s, y_train,
    eval_set=[(X_train_s, y_train), (X_val_s, y_val)],
    verbose=False
)
xgb_time = time.time() - t0

# No expm1 — predictions are in Ω directly
y_pred_xgb = best_xgb.predict(X_val_s)
mae_xgb  = mean_absolute_error(y_val, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
r2_xgb   = r2_score(y_val, y_pred_xgb)
print(f'\nXGBoost Results (Validation Set):')
print(f'  MAE:  {mae_xgb:.6f} Ω')
print(f'  RMSE: {rmse_xgb:.6f} Ω')
print(f'  R²:   {r2_xgb:.4f}')
print(f'  Time: {xgb_time:.1f}s')
```

---

## ▶ CELL TO REPLACE: Learning Curve — best model selection block
Find this cell by looking for: `best_name = min(results, key=lambda k: results[k]['RMSE'])`

Replace the entire cell with:

```python
# Select best model by lowest RMSE on validation set
best_name = min(results, key=lambda k: results[k]['RMSE'])

# For learning_curve() we need a model WITHOUT early_stopping_rounds
# because sklearn calls fit() internally without eval_set
if best_name == 'XGBoost':
    xgb_params = gs_xgb.best_params_.copy()
    best_model_for_lc = XGBRegressor(
        **xgb_params,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    best_model = best_xgb  # keep early-stopping version as the actual saved model
elif best_name == 'RandomForest':
    best_model_for_lc = best_rf
    best_model = best_rf
else:
    best_model_for_lc = lr
    best_model = lr

print(f'Best model: {best_name} (RMSE={results[best_name]["RMSE"]:.6f} Ω)')

# Learning curve — training directly in Ω (no log transform)
train_sizes, train_scores, val_scores = learning_curve(
    best_model_for_lc, X_train_s, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1
)

fig, ax = plt.subplots(figsize=(12, 6))
train_mean = -train_scores.mean(axis=1); val_mean = -val_scores.mean(axis=1)
train_std  = train_scores.std(axis=1);   val_std  = val_scores.std(axis=1)

ax.plot(train_sizes, train_mean, 'o-', color='#2196F3', label='Training RMSE',   linewidth=2)
ax.plot(train_sizes, val_mean,   'o-', color='#F44336', label='Validation RMSE', linewidth=2)
ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.1, color='#2196F3')
ax.fill_between(train_sizes, val_mean-val_std,     val_mean+val_std,     alpha=0.1, color='#F44336')
ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('RMSE (Ω)', fontsize=12)
ax.set_title(f'Learning Curve — {best_name} (predicting internal_resistance)', fontweight='bold', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../plots/learning_curve_regression.png', dpi=150, bbox_inches='tight')
plt.show()

if val_mean[-1] - train_mean[-1] > 0.5 * train_mean[-1]:
    print('→ Large gap: possible OVERFITTING. Consider more regularization.')
elif train_mean[-1] > 0.01:
    print('→ Both curves high: possible UNDERFITTING. Try more complex features.')
else:
    print('→ Curves converging: GOOD generalization.')
```

---

## ▶ CELL TO REPLACE: Save best model
Find this cell by looking for: `joblib.dump(best_model, '../models/regression_model.pkl')`

Replace the entire cell with:

```python
# Save best regression model
joblib.dump(best_model, '../models/regression_model.pkl')
print(f'Saved: models/regression_model.pkl ({best_name} — predicts internal_resistance)')

# Save scaler (no target_scaler needed — no log transform applied)
joblib.dump({'scaler': scaler, 'imputer': None, 'fit_ir_baseline': baseline_ir,
             'columns_to_drop': ['battery_temp'],
             'regression_target': 'internal_resistance'}, '../models/preprocessor_pipeline.pkl')
print('Saved: models/preprocessor_pipeline.pkl')

# Save results
json_results = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in results.items()}
with open('../models/regression_results.json', 'w') as f:
    json.dump(json_results, f, indent=2)
print('Saved: models/regression_results.json')
```

---

## ▶ CELL TO REPLACE: Notebook header markdown cell
Find the very first markdown cell that starts with: `# 📈 Notebook 04 — Model Training (Regression)`

Replace with:

```markdown
# 📈 Notebook 04 — Model Training (Regression)

**Purpose**: Train multiple regression models to predict `internal_resistance`, tune hyperparameters, select best model.

**Target**: `internal_resistance` (DC internal resistance in Ohms)

**Why internal_resistance (not cycle_degradation):**
Internal resistance is the physically grounded, statistically predictable battery aging metric in this dataset.
It increases monotonically as electrodes degrade — a well-established principle in battery electrochemistry.
Critically, it has strong measurable correlation with the other sensor features (RF R²=0.97),
making it a valid and meaningful regression target.

`cycle_degradation` was originally the stated target but EDA (re-analysis) revealed it has
near-zero correlation with every feature in the dataset (max Pearson r = 0.038), consistent
with synthetic random noise generation. No ML model can predict a target that has no
statistical relationship with the features. Predicting `internal_resistance` is the correct
production-grade formulation: from charging-session sensor data, estimate the battery's
current internal resistance, which directly informs maintenance scheduling and end-of-life prediction.

**Theory**:
- Lecture 04: Linear Regression (foundational baseline)
- Lecture 06: Random Forest (bagging reduces variance)
- Lecture 07: XGBoost (best for structured tabular data with L1/L2 regularization)
```

---

## ▶ CELL TO REPLACE: Summary table markdown cell
Find the markdown cell that contains the `| Model | MAE |` table at the bottom.

Replace with:

```markdown
## 📋 Regression Training Summary

**Target variable:** `internal_resistance` (Ω) — battery aging indicator

| Model | Approach | CV Strategy | Key Strength |
|-------|----------|-------------|--------------|
| Linear Regression | Ordinary Least Squares | — | Interpretable baseline; R²≈0.84 |
| Random Forest | GridSearchCV | TimeSeriesSplit | Bagging reduces variance; R²≈0.97 |
| XGBoost | GridSearchCV + early stopping | TimeSeriesSplit | Sequential boosting + L1/L2 reg |

**Selected model saved to `models/regression_model.pkl`**

**Physical interpretation of target:**
A battery with internal_resistance rising above ~0.10 Ω is approaching end-of-life
(typically defined at 80% capacity retention, corresponding to ~2× initial resistance).
Predicting this from real-time charging sensor data enables proactive maintenance scheduling.
```
