"""
BatteryPreprocessor — Reusable preprocessing class for the EV Battery ML Pipeline.

Applies all cleaning, engineering, and scaling steps identically for training and
inference. Ensures no data leakage: scaler is fit only on training data, and all
engineering baselines (like IR baseline) are computed from training data only.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import feature engineering functions
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.feature_engineer import engineer_all_features


class BatteryPreprocessor:
    """
    Preprocessing pipeline for EV battery charging data.
    
    Handles:
    - Dropping timestamp column
    - Feature engineering (delta IR, rolling SOC, thermal acceleration, etc.)
    - Dropping redundant features
    - Median imputation for missing values
    - Standard scaling (fit on training data only)
    - Log1p / expm1 target transformation for cycle_degradation
    
    Usage:
        # Training
        preprocessor = BatteryPreprocessor()
        X_train_scaled = preprocessor.fit_transform(X_train)
        preprocessor.save('models/')
        
        # Inference
        preprocessor = BatteryPreprocessor.load('models/')
        X_new_scaled = preprocessor.transform(X_new)
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize the preprocessor.
        
        Args:
            models_dir: Path to models directory. If provided and artifacts exist,
                       loads fitted scaler and feature columns from disk.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.fit_ir_baseline = None  # First internal_resistance from training set
        self.columns_to_drop = ['battery_temp']  # Determined in Notebook 03
        self.is_fitted = False
        
        if models_dir and os.path.exists(models_dir):
            self._try_load(models_dir)
    
    def _try_load(self, models_dir: str):
        """Try to load fitted artifacts from disk."""
        scaler_path = os.path.join(models_dir, 'preprocessor_pipeline.pkl')
        features_path = os.path.join(models_dir, 'feature_columns.json')
        config_path = os.path.join(models_dir, 'preprocessor_config.json')
        
        if os.path.exists(scaler_path):
            saved = joblib.load(scaler_path)
            self.scaler = saved['scaler']
            self.imputer = saved['imputer']
            self.fit_ir_baseline = saved['fit_ir_baseline']
            self.columns_to_drop = saved.get('columns_to_drop', ['battery_temp'])
            self.is_fitted = True
        
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
    
    def fit(self, X_train: pd.DataFrame):
        """
        Fit the preprocessor on training data.
        
        Computes:
        - IR baseline from first row of training data
        - Imputer fit on training features (after engineering)
        - Scaler fit on training features (after engineering + imputation)
        
        Args:
            X_train: Training features DataFrame (without target columns).
        """
        X = X_train.copy()
        
        # Drop timestamp if present
        if 'timestamp' in X.columns:
            X = X.drop(columns=['timestamp'])
        
        # Store the IR baseline from training data
        self.fit_ir_baseline = X['internal_resistance'].iloc[0]
        
        # Apply feature engineering
        X = engineer_all_features(X, self.fit_ir_baseline, self.columns_to_drop)
        
        # Select feature columns (if not set by RFE, use all)
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        
        X_selected = X[self.feature_columns]
        
        # Fit imputer
        self.imputer.fit(X_selected)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_selected),
            columns=self.feature_columns,
            index=X_selected.index
        )
        
        # Fit scaler
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using the fitted preprocessor.
        
        Steps:
        1. Drop timestamp if present
        2. Compute all engineered features (using training IR baseline)
        3. Select only the feature columns used during training
        4. Impute missing values with training medians
        5. Scale with training-fitted scaler
        
        Args:
            X: DataFrame with raw feature columns.
        
        Returns:
            Scaled numpy array ready for model prediction.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted. Call fit() first.")
        
        X = X.copy()
        
        # Drop timestamp if present
        if 'timestamp' in X.columns:
            X = X.drop(columns=['timestamp'])
        
        # Apply feature engineering using TRAINING baseline
        X = engineer_all_features(X, self.fit_ir_baseline, self.columns_to_drop)
        
        # Select only trained feature columns
        X_selected = X[self.feature_columns]
        
        # Impute and scale
        X_imputed = self.imputer.transform(X_selected)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training data and return transformed training data."""
        self.fit(X_train)
        return self.transform(X_train)
    
    def save(self, models_dir: str):
        """
        Save all fitted artifacts to disk.
        
        Saves:
        - preprocessor_pipeline.pkl: scaler, imputer, IR baseline, drop columns
        - feature_columns.json: list of selected feature column names
        """
        os.makedirs(models_dir, exist_ok=True)
        
        # Save scaler + imputer + config as a single pickle
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'fit_ir_baseline': self.fit_ir_baseline,
            'columns_to_drop': self.columns_to_drop
        }, os.path.join(models_dir, 'preprocessor_pipeline.pkl'))
        
        # Save feature columns as JSON
        with open(os.path.join(models_dir, 'feature_columns.json'), 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
    
    @classmethod
    def load(cls, models_dir: str) -> 'BatteryPreprocessor':
        """Load a fitted preprocessor from disk."""
        instance = cls(models_dir=models_dir)
        if not instance.is_fitted:
            raise ValueError(f"No fitted preprocessor found in {models_dir}")
        return instance
    
    @staticmethod
    def transform_target(y: pd.Series) -> np.ndarray:
        """Apply log1p transform to cycle_degradation target."""
        return np.log1p(y)
    
    @staticmethod
    def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
        """
        Convert log-scale predictions back to original cycle_degradation scale.
        
        Use this when reporting final metrics so they are in real battery
        degradation units, not log-space.
        """
        return np.expm1(y_log)
