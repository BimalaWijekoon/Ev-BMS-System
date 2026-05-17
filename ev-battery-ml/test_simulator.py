#!/usr/bin/env python
"""Quick test of simulator module."""

from src.simulator import BatterySimulator

sim = BatterySimulator('data/nev_battery_charging.csv')
print('✅ Simulator initialized successfully')

state = sim.get_initial_state()
print(f'✅ Initial state: RAW SOC={state["raw"]["SOC"]:.1f}%, ENGINEERED features={len(state["engineered"])}')

state = sim.step(state, 1.0)
print(f'✅ After 1 step: RAW SOC={state["raw"]["SOC"]:.1f}%, ENGINEERED features={len(state["engineered"])}')

limits = sim.get_feature_limits()
print(f'✅ Feature limits loaded: {len(limits)} engineered features')

# Check what engineered features we have
eng_keys = list(state['engineered'].keys())
print(f'\n📊 Engineered Features ({len(eng_keys)}):')
for i, key in enumerate(eng_keys, 1):
    val = state['engineered'][key]
    print(f'   {i:2d}. {key:30s} = {val:.6f}' if isinstance(val, float) else f'   {i:2d}. {key:30s} = {val}')

print('\n🎉 All tests passed!')

