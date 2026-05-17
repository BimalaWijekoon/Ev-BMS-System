"""
Battery Insights Engine
Converts raw predictions into actionable business intelligence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class BatteryInsights:
    """Analyzes battery data and predictions to generate meaningful insights."""
    
    def __init__(self, csv_path: str = 'data/nev_battery_charging.csv'):
        """
        Initialize insights engine with historical data.
        
        Args:
            csv_path: Path to battery charging CSV
        """
        self.df = pd.read_csv(csv_path)
        self._compute_fleet_stats()
    
    def _compute_fleet_stats(self):
        """Compute baseline statistics from entire fleet."""
        self.fleet_stats = {
            'ir_mean': self.df['internal_resistance'].mean(),
            'ir_std': self.df['internal_resistance'].std(),
            'ir_min': self.df['internal_resistance'].min(),
            'ir_max': self.df['internal_resistance'].max(),
            'temp_mean': self.df['battery_temp'].mean(),
            'over_temp_rate': (self.df['over_temp_flag'].sum() / len(self.df)) * 100,
            'over_voltage_rate': (self.df['over_voltage_flag'].sum() / len(self.df)) * 100,
            'avg_charging_efficiency': self.df['charging_efficiency'].mean(),
            'total_cycles': len(self.df),
        }
        
        # IR degradation trajectory (how IR changes with SOC progression)
        self.df['soc_bin'] = pd.cut(self.df['SOC'], bins=20, labels=False)
        self.ir_by_soc = self.df.groupby('soc_bin')['internal_resistance'].agg(['mean', 'std', 'count'])
    
    def assess_battery_health(self, current_ir: float, current_over_temp_flag: int, 
                             current_over_volt_flag: int) -> Dict:
        """
        Assess current battery health based on predictions.
        
        Args:
            current_ir: Predicted internal resistance (Ω)
            current_over_temp_flag: Over-temperature flag (0=safe, 1=warning)
            current_over_volt_flag: Over-voltage flag (0=safe, 1=warning)
        
        Returns:
            dict with health_score, status, and alerts
        """
        health_scores = []
        alerts = []
        
        # 1. IR-based health (normalized to 0-100)
        ir_percentile = (current_ir - self.fleet_stats['ir_min']) / \
                        (self.fleet_stats['ir_max'] - self.fleet_stats['ir_min']) * 100
        ir_health = max(0, 100 - ir_percentile)  # Lower IR = higher health
        health_scores.append(('IR Condition', ir_health, 0.4))
        
        if current_ir > self.fleet_stats['ir_mean'] + 2 * self.fleet_stats['ir_std']:
            alerts.append({
                'type': 'HIGH_DEGRADATION',
                'severity': 'CRITICAL',
                'message': f'IR is {(current_ir / self.fleet_stats["ir_mean"] - 1) * 100:.1f}% above fleet average. Severe degradation detected.',
                'value': current_ir
            })
        elif current_ir > self.fleet_stats['ir_mean'] + self.fleet_stats['ir_std']:
            alerts.append({
                'type': 'DEGRADATION_WARNING',
                'severity': 'WARNING',
                'message': f'IR is {(current_ir / self.fleet_stats["ir_mean"] - 1) * 100:.1f}% above fleet average. Monitor closely.',
                'value': current_ir
            })
        
        # 2. Temperature safety
        temp_health = 100 if current_over_temp_flag == 0 else 40
        health_scores.append(('Thermal Safety', temp_health, 0.35))
        
        if current_over_temp_flag == 1:
            alerts.append({
                'type': 'OVER_TEMP',
                'severity': 'CRITICAL',
                'message': 'Battery temperature exceeds safe threshold. Risk of thermal damage.',
                'value': 'ACTIVE'
            })
        
        # 3. Voltage safety
        volt_health = 100 if current_over_volt_flag == 0 else 40
        health_scores.append(('Voltage Safety', volt_health, 0.25))
        
        if current_over_volt_flag == 1:
            alerts.append({
                'type': 'OVER_VOLTAGE',
                'severity': 'CRITICAL',
                'message': 'Battery voltage exceeds safe threshold. Risk of cell damage.',
                'value': 'ACTIVE'
            })
        
        # Weighted health score
        overall_health = sum(score * weight for _, score, weight in health_scores) / \
                        sum(weight for _, _, weight in health_scores)
        
        # Status determination
        if overall_health >= 80:
            status = '🟢 HEALTHY'
        elif overall_health >= 60:
            status = '🟡 DEGRADING'
        elif overall_health >= 40:
            status = '🔴 CRITICAL'
        else:
            status = '⚫ END_OF_LIFE'
        
        return {
            'overall_health': overall_health,
            'status': status,
            'component_scores': health_scores,
            'alerts': alerts,
        }
    
    def estimate_remaining_useful_life(self, current_ir: float, ir_history: List[float] = None,
                                      ir_max_threshold: float = 0.15) -> Dict:
        """
        Estimate Remaining Useful Life (RUL) based on IR degradation.
        
        Args:
            current_ir: Current internal resistance (Ω)
            ir_history: List of recent IR predictions (optional, for trend analysis)
            ir_max_threshold: IR threshold beyond which battery is considered EOL
        
        Returns:
            dict with RUL estimate and degradation rate
        """
        rul_info = {
            'current_ir': current_ir,
            'ir_threshold': ir_max_threshold,
            'ir_remaining': max(0, ir_max_threshold - current_ir),
        }
        
        if ir_history and len(ir_history) > 2:
            # Calculate degradation rate from history
            ir_history = np.array(ir_history)
            time_steps = np.arange(len(ir_history))
            
            # Linear regression: IR vs time
            coeffs = np.polyfit(time_steps, ir_history, 1)
            degradation_rate = coeffs[0]  # slope
            
            rul_info['degradation_rate'] = degradation_rate
            rul_info['degradation_rate_unit'] = 'Ω per cycle'
            
            if degradation_rate > 0:
                # Estimate cycles until threshold
                cycles_remaining = rul_info['ir_remaining'] / degradation_rate
                rul_info['estimated_cycles_remaining'] = max(0, int(cycles_remaining))
                
                # Conservative estimate (assume degradation accelerates)
                rul_info['estimated_cycles_remaining_conservative'] = \
                    max(0, int(cycles_remaining * 0.7))
            else:
                rul_info['estimated_cycles_remaining'] = None
                rul_info['estimated_cycles_remaining_conservative'] = None
        else:
            # No history - use fleet average degradation
            fleet_degradation = self.fleet_stats['ir_std'] / self.fleet_stats['total_cycles']
            rul_info['degradation_rate'] = fleet_degradation
            rul_info['degradation_rate_unit'] = 'Ω per cycle (fleet avg)'
            
            if fleet_degradation > 0:
                cycles_remaining = rul_info['ir_remaining'] / fleet_degradation
                rul_info['estimated_cycles_remaining'] = max(0, int(cycles_remaining))
                rul_info['estimated_cycles_remaining_conservative'] = \
                    max(0, int(cycles_remaining * 0.7))
        
        return rul_info
    
    def compare_to_fleet(self, current_ir: float, current_efficiency: float,
                        current_over_temp_prob: float) -> Dict:
        """
        Compare battery to fleet averages.
        
        Args:
            current_ir: Current IR prediction
            current_efficiency: Current charging efficiency
            current_over_temp_prob: Over-temperature probability (0-1)
        
        Returns:
            dict with percentile rankings and comparisons
        """
        ir_percentile = (current_ir - self.fleet_stats['ir_min']) / \
                       (self.fleet_stats['ir_max'] - self.fleet_stats['ir_min']) * 100
        
        return {
            'ir_vs_fleet': {
                'current': current_ir,
                'fleet_average': self.fleet_stats['ir_mean'],
                'fleet_std': self.fleet_stats['ir_std'],
                'percentile': ir_percentile,
                'status': 'BETTER' if current_ir < self.fleet_stats['ir_mean'] else 'WORSE',
                'deviation_sigmas': (current_ir - self.fleet_stats['ir_mean']) / self.fleet_stats['ir_std'],
            },
            'efficiency_vs_fleet': {
                'current': current_efficiency,
                'fleet_average': self.fleet_stats['avg_charging_efficiency'],
                'percentile': (current_efficiency / self.fleet_stats['avg_charging_efficiency']) * 100,
                'status': 'ABOVE_AVERAGE' if current_efficiency > self.fleet_stats['avg_charging_efficiency'] else 'BELOW_AVERAGE',
            },
            'over_temp_risk_vs_fleet': {
                'current_probability': current_over_temp_prob * 100,
                'fleet_occurrence_rate': self.fleet_stats['over_temp_rate'],
                'risk_level': 'HIGH' if current_over_temp_prob * 100 > self.fleet_stats['over_temp_rate'] else 'LOW',
            }
        }
    
    def get_recommendations(self, health_score: float, rul_info: Dict,
                           alerts: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            health_score: Overall battery health score (0-100)
            rul_info: RUL estimation dict
            alerts: List of active alerts
        
        Returns:
            list of recommendation strings
        """
        recommendations = []
        
        # Based on health score
        if health_score >= 80:
            recommendations.append("✅ Battery is in good condition. Continue normal operation with routine monitoring.")
        elif health_score >= 60:
            recommendations.append("⚠️ Battery shows signs of degradation. Increase monitoring frequency to weekly.")
            recommendations.append("💡 Consider scheduling maintenance within the next 2-3 weeks.")
        elif health_score >= 40:
            recommendations.append("🔴 Battery is degrading rapidly. Daily monitoring required.")
            recommendations.append("⚡ Reduce charging power if possible to slow degradation.")
            recommendations.append("📞 Contact maintenance team for urgent assessment.")
        else:
            recommendations.append("🚨 Battery has reached end-of-life. STOP charging immediately.")
            recommendations.append("📞 Replace battery as soon as possible.")
        
        # Based on RUL
        if 'estimated_cycles_remaining_conservative' in rul_info:
            cycles = rul_info['estimated_cycles_remaining_conservative']
            if cycles is not None:
                if cycles < 10:
                    recommendations.append(f"⏰ RUL estimate: {cycles} cycles. Battery retirement imminent.")
                elif cycles < 50:
                    recommendations.append(f"📊 RUL estimate: {cycles} cycles. Plan replacement soon.")
                else:
                    recommendations.append(f"📈 RUL estimate: {cycles} cycles. Monitor degradation trend.")
        
        # Based on alerts
        if any(alert['severity'] == 'CRITICAL' for alert in alerts):
            recommendations.append("🚨 CRITICAL ALERTS ACTIVE: Take immediate action.")
        
        return recommendations
    
    def get_fleet_summary(self) -> Dict:
        """Return summary statistics of the entire fleet."""
        return {
            'total_batteries_monitored': self.fleet_stats['total_cycles'],
            'avg_ir': self.fleet_stats['ir_mean'],
            'ir_range': (self.fleet_stats['ir_min'], self.fleet_stats['ir_max']),
            'over_temp_occurrence_rate': f"{self.fleet_stats['over_temp_rate']:.1f}%",
            'over_voltage_occurrence_rate': f"{self.fleet_stats['over_voltage_rate']:.1f}%",
            'avg_efficiency': f"{self.fleet_stats['avg_charging_efficiency']:.1%}",
        }
