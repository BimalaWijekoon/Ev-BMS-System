"""
Feature Engineering Module for EV Battery Charging ML Pipeline.

Contains standalone functions for all feature engineering steps.
Each function accepts a DataFrame and returns the DataFrame with new columns added.
These functions are called inside BatteryPreprocessor.transform().
"""

import numpy as np
import pandas as pd


def add_delta_internal_resistance(df: pd.DataFrame, baseline_ir: float) -> pd.DataFrame:
    """
    Compute the change in internal resistance from the training baseline.
    
    This represents cumulative resistance growth — a direct proxy for battery aging.
    Battery physics: internal resistance increases as batteries age due to electrode
    degradation and electrolyte decomposition. The change from baseline is a universal
    aging signal regardless of cell chemistry.
    
    Args:
        df: DataFrame with 'internal_resistance' column.
        baseline_ir: The first internal_resistance value from the training set.
                     During inference, this must be the training baseline, NOT the
                     inference data's first value.
    
    Returns:
        DataFrame with 'delta_internal_resistance' column added.
    """
    df = df.copy()
    df['delta_internal_resistance'] = df['internal_resistance'] - baseline_ir
    return df


def add_soc_range_rolling(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute a rolling window of SOC range: max(SOC) - min(SOC) within the window.
    
    This captures how aggressively the battery is being cycled. Batteries cycled
    between 0-100% SOC degrade faster than those kept in the 20-80% range. This
    feature encodes cycling aggressiveness — a key degradation driver not captured
    by any single-point SOC reading.
    
    Args:
        df: DataFrame with 'SOC' column.
        window: Rolling window size (default 10).
    
    Returns:
        DataFrame with 'soc_range_rolling' column added.
        First `window` NaN values are filled with the column median.
    """
    df = df.copy()
    df['soc_range_rolling'] = (
        df['SOC'].rolling(window=window).max() - df['SOC'].rolling(window=window).min()
    )
    # Fill NaN values from the first window with the median of the column
    median_val = df['soc_range_rolling'].median()
    df['soc_range_rolling'] = df['soc_range_rolling'].fillna(median_val)
    return df


def add_thermal_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiply dT_dt (rate of temperature change) by battery_temp.
    
    High temperature combined with rapid temperature change is more damaging than
    either alone. This is a compound feature — combining two features to create a
    higher-information feature (like combining lat+lon into distance).
    
    High dT_dt at 30°C is very different from high dT_dt at 1000°C in terms of
    degradation impact.
    
    Args:
        df: DataFrame with 'dT_dt' and 'battery_temp' columns.
    
    Returns:
        DataFrame with 'thermal_acceleration' column added.
    """
    df = df.copy()
    df['thermal_acceleration'] = df['dT_dt'] * df['battery_temp']
    return df


def add_voltage_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute action_voltage / terminal_voltage ratio.
    
    Values close to 1.0 mean the battery is delivering what was asked.
    Values far from 1.0 indicate voltage deviation stress — a signal of battery
    impedance and health not captured by either column individually.
    
    Args:
        df: DataFrame with 'action_voltage' and 'terminal_voltage' columns.
    
    Returns:
        DataFrame with 'voltage_efficiency' column added.
    """
    df = df.copy()
    df['voltage_efficiency'] = df['action_voltage'] / (df['terminal_voltage'] + 1e-6)
    return df


def add_polynomial_features(
    df: pd.DataFrame,
    columns_to_square: list = None
) -> pd.DataFrame:
    """
    Add squared terms for specified columns to capture non-linear relationships.
    
    Battery degradation follows non-linear chemistry — it accelerates exponentially
    at high resistance and high temperature, not linearly. Squared terms allow
    linear models to capture this curve.
    
    Args:
        df: DataFrame containing the columns to square.
        columns_to_square: List of column names. Defaults to
                          ['internal_resistance', 'thermal_stress_index', 'aging_indicator'].
    
    Returns:
        DataFrame with squared columns appended (named '{col}_sq').
    """
    if columns_to_square is None:
        columns_to_square = ['internal_resistance', 'thermal_stress_index', 'aging_indicator']
    
    df = df.copy()
    for col in columns_to_square:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
    return df


def drop_redundant_features(df: pd.DataFrame, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Drop redundant or highly correlated features.
    
    The drop list is determined in Notebook 03 feature selection analysis.
    Default columns to drop are based on high correlation analysis:
    - battery_temp: >0.95 correlation with thermal_stress_index (keep the normalized one)
    - timestamp: if still present, contains no predictive value
    
    Args:
        df: DataFrame with feature columns.
        columns_to_drop: List of column names to drop. If None, uses defaults.
    
    Returns:
        DataFrame without the dropped columns.
    """
    if columns_to_drop is None:
        columns_to_drop = ['battery_temp']
    
    df = df.copy()
    cols_to_drop = [c for c in columns_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def engineer_all_features(df: pd.DataFrame, baseline_ir: float, columns_to_drop: list = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.
    
    This is the main entry point for the preprocessing pipeline.
    
    Args:
        df: Raw DataFrame (after timestamp removal).
        baseline_ir: Training set's first internal_resistance value.
        columns_to_drop: Features to remove after engineering.
    
    Returns:
        Fully engineered DataFrame.
    """
    df = add_delta_internal_resistance(df, baseline_ir)
    df = add_soc_range_rolling(df, window=10)
    df = add_thermal_acceleration(df)
    df = add_voltage_efficiency(df)
    df = add_polynomial_features(df)
    df = drop_redundant_features(df, columns_to_drop)
    return df
