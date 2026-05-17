"""
Battery Charging Simulator
Generates realistic feature progressions based on training data statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_engineer import engineer_all_features


class BatterySimulator:
    """Simulates battery charging from 0-100% SOC with realistic feature progression."""
    
    def __init__(self, data_path='data/nev_battery_charging.csv'):
        """
        Initialize simulator with statistical patterns from training data.
        
        Args:
            data_path: Path to CSV file with historical charging data
        """
        self.df = pd.read_csv(data_path)
        self._analyze_patterns()
        
        # Load RFE-selected features (what models expect)
        import json
        try:
            with open('models/feature_columns.json') as f:
                self.rfe_features = json.load(f)
        except:
            self.rfe_features = None  # Will use all features if not found
        
        self.baseline_ir = self.df['internal_resistance'].iloc[0]
    
    def _analyze_patterns(self):
        """Extract patterns from training data for realistic simulation."""
        # Group by SOC bins to understand feature evolution during charging
        self.df['soc_bin'] = pd.cut(self.df['SOC'], bins=10, labels=False)
        
        # Calculate statistics for each SOC level
        self.soc_stats = {}
        for bin_idx in range(10):
            bin_data = self.df[self.df['soc_bin'] == bin_idx]
            if len(bin_data) > 0:
                self.soc_stats[bin_idx] = {
                    'terminal_voltage': bin_data['terminal_voltage'].mean(),
                    'battery_temp': bin_data['battery_temp'].mean(),
                    'battery_current': bin_data['battery_current'].mean(),
                    'internal_resistance': bin_data['internal_resistance'].mean(),
                    'charging_efficiency': bin_data['charging_efficiency'].mean(),
                }
        
        # Get min/max for normalization
        self.feature_ranges = {}
        for col in ['SOC', 'SOH', 'terminal_voltage', 'battery_current', 'battery_temp',
                    'ambient_temp', 'internal_resistance', 'charging_efficiency']:
            if col in self.df.columns:
                self.feature_ranges[col] = {
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'mean': self.df[col].mean(),
                }
    
    def get_initial_state(self):
        """
        Return initial battery state at 0% SOC.
        
        Returns:
            dict with 'raw' and 'engineered' keys:
            - raw: 17 raw features (SOC, temp, current, etc.)
            - engineered: 15 RFE-selected engineered features (for models)
        """
        raw_state = {
            'SOC': 1.0,
            'SOH': 0.95,
            'terminal_voltage': 2.8,
            'battery_current': 20.0,
            'battery_temp': 25.0,
            'ambient_temp': 25.0,
            'internal_resistance': 0.085,
            'action_current': 15.0,
            'action_voltage': 2.8,
            'dT_dt': 0.0,
            'dV_dt': 0.0,
            'soc_delta': 0.05,
            'thermal_stress_index': 0.0,
            'aging_indicator': 0.2,
            'charging_efficiency': 0.85,
            'charging_time': 100.0,
            'balancing_time': 10.0,
        }
        
        # Convert to DataFrame and apply feature engineering
        df = pd.DataFrame([raw_state])
        df_engineered = engineer_all_features(df, self.baseline_ir, columns_to_drop=['battery_temp'])
        
        # Select only RFE-selected features
        if self.rfe_features:
            engineered_dict = {}
            for feat in self.rfe_features:
                if feat in df_engineered.columns:
                    engineered_dict[feat] = df_engineered[feat].iloc[0]
        else:
            engineered_dict = df_engineered.iloc[0].to_dict()
        
        return {
            'raw': raw_state,
            'engineered': engineered_dict
        }
    
    def step(self, current_state, soc_increment=1.0, temp_adjustment=0.0):
        """
        Advance simulation by one step.
        
        Args:
            current_state: State dict with 'raw' and 'engineered' keys
            soc_increment: How much to increase SOC (%)
            temp_adjustment: Manual temperature adjustment (°C)
        
        Returns:
            Updated state dict with 'raw' and 'engineered' keys
        """
        # Work with raw state
        raw_state = current_state['raw'].copy()
        
        # Update SOC (primary driver)
        raw_state['SOC'] = min(100.0, raw_state['SOC'] + soc_increment)
        
        # Interpolate features based on new SOC from learned patterns
        soc_bin = int((raw_state['SOC'] / 100.0) * 9)
        soc_bin = max(0, min(8, soc_bin))
        
        if soc_bin in self.soc_stats:
            stats = self.soc_stats[soc_bin]
            alpha = 0.3
            raw_state['terminal_voltage'] = (
                raw_state['terminal_voltage'] * (1 - alpha) +
                stats['terminal_voltage'] * alpha
            )
            raw_state['battery_current'] = (
                raw_state['battery_current'] * (1 - alpha) +
                stats['battery_current'] * alpha
            )
            raw_state['internal_resistance'] = (
                raw_state['internal_resistance'] * (1 - alpha) +
                stats['internal_resistance'] * alpha
            )
        
        # Temperature dynamics
        current_effect = (raw_state['battery_current'] / 30.0) * 0.5
        raw_state['battery_temp'] += current_effect
        raw_state['battery_temp'] = max(25.0, min(50.0, raw_state['battery_temp']))
        
        if temp_adjustment != 0.0:
            raw_state['battery_temp'] += temp_adjustment
        
        # SOH slowly degrades
        raw_state['SOH'] = max(0.5, raw_state['SOH'] - 0.001)
        
        # Update derived features
        raw_state['dT_dt'] = current_effect
        raw_state['dV_dt'] = (raw_state['terminal_voltage'] - 2.8) / 10.0
        raw_state['soc_delta'] = soc_increment / 100.0
        raw_state['thermal_stress_index'] = (raw_state['battery_temp'] - 25.0) * raw_state['battery_current'] / 100.0
        raw_state['aging_indicator'] = (1.0 - raw_state['SOH']) * 0.5
        raw_state['charging_time'] = (100.0 - raw_state['SOC']) * 30.0
        raw_state['balancing_time'] = max(0.0, 10.0 - raw_state['SOC'] / 10.0)
        
        # Convert to DataFrame and apply feature engineering
        df = pd.DataFrame([raw_state])
        df_engineered = engineer_all_features(df, self.baseline_ir, columns_to_drop=['battery_temp'])
        
        # Select only RFE-selected features
        if self.rfe_features:
            engineered_dict = {}
            for feat in self.rfe_features:
                if feat in df_engineered.columns:
                    engineered_dict[feat] = df_engineered[feat].iloc[0]
        else:
            engineered_dict = df_engineered.iloc[0].to_dict()
        
        return {
            'raw': raw_state,
            'engineered': engineered_dict
        }
    
    def get_feature_limits(self):
        """Return min/max ranges for RFE-selected features."""
        if not self.rfe_features:
            return {}
        
        limits = {}
        for feat in self.rfe_features:
            if feat in self.df.columns:
                limits[feat] = (
                    float(self.df[feat].min()),
                    float(self.df[feat].max())
                )
            else:
                # Engineered features - provide reasonable defaults
                limits[feat] = (0, 1)
        
        return limits
