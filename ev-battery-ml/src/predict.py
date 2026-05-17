"""
Inference module for EV Battery ML Pipeline.
Loads all saved models and config, runs predictions, returns structured results.
Regression target: internal_resistance (Ω)
Classification target: over_temp_flag (ML) + over_voltage_flag (rule-based)
"""
import json, os, sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
sys.path.insert(0, BASE_DIR)
from src.preprocessor import BatteryPreprocessor

def load_models(models_dir=None):
    """Load all saved models and config from models/ directory."""
    md = models_dir or MODELS_DIR
    loaded = {}
    
    try:
        loaded['preprocessor'] = BatteryPreprocessor.load(md)
    except Exception as e:
        raise RuntimeError(f"Failed to load preprocessor: {str(e)}")
    
    try:
        loaded['regression_model'] = joblib.load(os.path.join(md, 'regression_model.pkl'))
    except Exception as e:
        raise RuntimeError(f"Failed to load regression model: {str(e)}")
    
    try:
        loaded['classification_model_temp'] = joblib.load(os.path.join(md, 'classification_model_temp.pkl'))
    except Exception as e:
        raise RuntimeError(f"Failed to load classification model: {str(e)}")
    
    try:
        with open(os.path.join(md, 'classification_config.json'), 'r') as f:
            loaded['config'] = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load classification config: {str(e)}")
    
    try:
        with open(os.path.join(md, 'feature_columns.json'), 'r') as f:
            loaded['feature_columns'] = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load feature columns: {str(e)}")
    
    return loaded

def check_model_health(models_dir=None) -> dict:
    """
    Check the health status of all loaded models.
    
    Returns:
        Dict with health status of each component.
    """
    md = models_dir or MODELS_DIR
    health = {
        'status': 'healthy',
        'errors': [],
        'components': {}
    }
    
    # Check preprocessor
    try:
        pp = BatteryPreprocessor.load(md)
        health['components']['preprocessor'] = {
            'status': 'ok',
            'fitted': pp.is_fitted,
            'features': len(pp.feature_columns) if pp.feature_columns else 0,
            'has_scaler': pp.scaler is not None,
            'has_imputer': pp.imputer is not None
        }
    except Exception as e:
        health['components']['preprocessor'] = {'status': 'error', 'error': str(e)}
        health['errors'].append(f"Preprocessor: {str(e)}")
        health['status'] = 'unhealthy'
    
    # Check regression model
    try:
        reg_path = os.path.join(md, 'regression_model.pkl')
        if os.path.exists(reg_path):
            reg = joblib.load(reg_path)
            health['components']['regression_model'] = {
                'status': 'ok',
                'type': type(reg).__name__,
                'file_exists': True
            }
        else:
            health['components']['regression_model'] = {'status': 'missing', 'file_exists': False}
            health['errors'].append("Regression model file not found")
            health['status'] = 'unhealthy'
    except Exception as e:
        health['components']['regression_model'] = {'status': 'error', 'error': str(e)}
        health['errors'].append(f"Regression model: {str(e)}")
        health['status'] = 'unhealthy'
    
    # Check classification model
    try:
        cls_path = os.path.join(md, 'classification_model_temp.pkl')
        if os.path.exists(cls_path):
            cls = joblib.load(cls_path)
            health['components']['classification_model'] = {
                'status': 'ok',
                'type': type(cls).__name__,
                'file_exists': True
            }
        else:
            health['components']['classification_model'] = {'status': 'missing', 'file_exists': False}
            health['errors'].append("Classification model file not found")
            health['status'] = 'unhealthy'
    except Exception as e:
        health['components']['classification_model'] = {'status': 'error', 'error': str(e)}
        health['errors'].append(f"Classification model: {str(e)}")
        health['status'] = 'unhealthy'
    
    # Check config files
    config_checks = [
        ('classification_config.json', 'classification_config'),
        ('feature_columns.json', 'feature_columns')
    ]
    for fname, name in config_checks:
        try:
            fpath = os.path.join(md, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r') as f:
                    content = json.load(f)
                health['components'][name] = {
                    'status': 'ok',
                    'file_exists': True,
                    'size': len(str(content))
                }
            else:
                health['components'][name] = {'status': 'missing', 'file_exists': False}
                health['errors'].append(f"{fname} not found")
                health['status'] = 'unhealthy'
        except Exception as e:
            health['components'][name] = {'status': 'error', 'error': str(e)}
            health['errors'].append(f"{name}: {str(e)}")
            health['status'] = 'unhealthy'
    
    return health

def predict_single(raw_input_dict: dict, loaded_models: dict = None) -> dict:
    """
    Run prediction for a single battery state observation.
    
    Args:
        raw_input_dict: Dictionary of feature name -> value pairs.
        loaded_models: Pre-loaded models dict (from load_models). Loads if None.
    
    Returns:
        Dict with keys: internal_resistance, over_temp_probability,
        over_temp_flag, over_voltage_flag
    """
    if loaded_models is None:
        loaded_models = load_models()
    
    pp = loaded_models['preprocessor']
    reg = loaded_models['regression_model']
    cls = loaded_models['classification_model_temp']
    config = loaded_models['config']
    
    # Build single-row DataFrame
    df = pd.DataFrame([raw_input_dict])
    
    # Ensure all needed raw columns exist with defaults
    required = ['SOC','SOH','terminal_voltage','battery_current','battery_temp',
        'ambient_temp','internal_resistance','action_current','action_voltage',
        'dT_dt','dV_dt','soc_delta','thermal_stress_index','aging_indicator',
        'charging_efficiency','charging_time','balancing_time']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
    
    # Transform
    X_scaled = pp.transform(df)
    
    # Regression prediction — internal_resistance in Ω (no log transform)
    ir_pred = float(reg.predict(X_scaled)[0])
    
    # Classification prediction
    probs = cls.predict_proba(X_scaled)[:,1]
    threshold = config.get('temp_threshold', 0.5)
    over_temp_prob = float(probs[0])
    over_temp_flag = int(probs[0] >= threshold)
    
    # Voltage rule-based fallback
    av = raw_input_dict.get('action_voltage', 0)
    tv = raw_input_dict.get('terminal_voltage', 0)
    over_voltage_flag = int(av > 4.15 or tv > 4.18)
    
    return {
        'internal_resistance': ir_pred,
        'over_temp_probability': over_temp_prob,
        'over_temp_flag': over_temp_flag,
        'over_voltage_flag': over_voltage_flag
    }

def predict_batch(df: pd.DataFrame, loaded_models: dict = None) -> pd.DataFrame:
    """
    Run predictions for a batch of observations.
    
    Args:
        df: DataFrame with raw feature columns.
        loaded_models: Pre-loaded models dict. Loads if None.
    
    Returns:
        Original DataFrame with prediction columns appended.
    """
    if loaded_models is None:
        loaded_models = load_models()
    
    pp = loaded_models['preprocessor']
    reg = loaded_models['regression_model']
    cls = loaded_models['classification_model_temp']
    config = loaded_models['config']
    threshold = config.get('temp_threshold', 0.5)
    
    df_out = df.copy()
    
    # Drop targets if present
    for t in ['cycle_degradation','over_temp_flag','over_voltage_flag','internal_resistance']:
        if t in df_out.columns:
            df_out = df_out.drop(columns=[t])
    
    X_scaled = pp.transform(df_out)
    
    # Regression — internal_resistance in Ω (no log transform)
    df_out['pred_internal_resistance'] = reg.predict(X_scaled)
    
    # Classification
    probs = cls.predict_proba(X_scaled)[:,1]
    df_out['pred_over_temp_prob'] = probs
    df_out['pred_over_temp_flag'] = (probs >= threshold).astype(int)
    
    # Voltage rule
    av = df_out.get('action_voltage', pd.Series([0]*len(df_out)))
    tv = df_out.get('terminal_voltage', pd.Series([0]*len(df_out)))
    df_out['pred_over_voltage_flag'] = ((av > 4.15) | (tv > 4.18)).astype(int)
    
    return df_out
