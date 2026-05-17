#!/usr/bin/env python
"""Quick test of insights module."""

from src.insights import BatteryInsights

# Initialize
insights = BatteryInsights('data/nev_battery_charging.csv')
print('✅ Insights engine initialized')

# Test 1: Health assessment
health = insights.assess_battery_health(0.085, 0, 0)
print(f'✅ Health assessment: {health["overall_health"]:.1f}% - {health["status"]}')

# Test 2: RUL estimation
rul = insights.estimate_remaining_useful_life(0.085)
print(f'✅ RUL estimation: {rul.get("estimated_cycles_remaining", "N/A")} cycles remaining')

# Test 3: Fleet comparison
comparison = insights.compare_to_fleet(0.085, 0.85, 0.1)
print(f'✅ Fleet comparison: IR percentile = {comparison["ir_vs_fleet"]["percentile"]:.1f}%')

# Test 4: Recommendations
recommendations = insights.get_recommendations(health['overall_health'], rul, health['alerts'])
print(f'✅ Recommendations: {len(recommendations)} items')

# Test 5: Fleet summary
fleet = insights.get_fleet_summary()
print(f'✅ Fleet summary: {fleet["total_batteries_monitored"]} batteries monitored')

print('\n🎉 All tests passed!')
