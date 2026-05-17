"""
Fix script to regenerate the preprocessor with proper imputer.
Loads training data and refits the preprocessor.
"""
import pandas as pd
import os
from src.preprocessor import BatteryPreprocessor

# Load training data
data_path = 'data/nev_battery_charging.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Training data not found at {data_path}")

print(f"Loading training data from {data_path}...")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Get feature columns from saved config
import json
with open('models/feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

print(f"Using {len(feature_cols)} feature columns")

# Prepare training data (exclude targets)
X_train = df.copy()
targets = ['cycle_degradation', 'over_temp_flag', 'over_voltage_flag']
for t in targets:
    if t in X_train.columns:
        X_train = X_train.drop(columns=[t])

print(f"Training features shape: {X_train.shape}")

# Create and fit preprocessor
print("Creating and fitting preprocessor...")
preprocessor = BatteryPreprocessor()
preprocessor.feature_columns = feature_cols
preprocessor.fit(X_train)

print(f"Preprocessor fitted!")
print(f"  - Scaler: {preprocessor.scaler}")
print(f"  - Imputer: {preprocessor.imputer}")
print(f"  - Features: {len(preprocessor.feature_columns)}")
print(f"  - Fitted: {preprocessor.is_fitted}")

# Save
print("Saving preprocessor...")
preprocessor.save('models/')
print("✅ Preprocessor saved successfully!")
