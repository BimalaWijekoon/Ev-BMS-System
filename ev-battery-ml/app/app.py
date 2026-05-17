"""
EV Battery Intelligence Predictor — Gradio Web Interface
Predicts internal resistance (aging proxy) and safety flags.
Two tabs: Single Prediction and Batch CSV Prediction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gradio as gr
import pandas as pd
import numpy as np
import tempfile

from src.predict import load_models, predict_single, predict_batch

# Load models at startup
print("Loading models...")
try:
    MODELS = load_models()
    print("Models loaded successfully!")
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Run notebooks 01-06 first to generate model artifacts.")
    MODELS_LOADED = False
    MODELS = None


def single_predict(soc, soh, terminal_voltage, battery_current, battery_temp,
                   ambient_temp, action_current,
                   action_voltage, charging_efficiency, charging_time):
    """Run single prediction and return formatted results."""
    if not MODELS_LOADED:
        return ("⚠️ Models not loaded", "⚠️ Models not loaded", "⚠️ Models not loaded", "⚠️ Models not loaded")
    
    raw_input = {
        'SOC': soc, 'SOH': soh, 'terminal_voltage': terminal_voltage,
        'battery_current': battery_current, 'battery_temp': battery_temp,
        'ambient_temp': ambient_temp,
        'action_current': action_current, 'action_voltage': action_voltage,
        'dT_dt': 0.0, 'dV_dt': 0.0, 'soc_delta': 0.0,
        'thermal_stress_index': min(battery_temp / 1129.0, 1.0),
        'aging_indicator': 1.0 - soh,
        'charging_efficiency': charging_efficiency,
        'charging_time': charging_time,
        'balancing_time': 15.0
    }
    
    result = predict_single(raw_input, MODELS)
    
    ir = result['internal_resistance']
    temp_prob = result['over_temp_probability'] * 100
    temp_status = "🟢 SAFE — Normal Temperature" if result['over_temp_flag'] == 0 else "🔴 WARNING — Over-Temperature Risk!"
    volt_status = "🟢 SAFE — Normal Voltage" if result['over_voltage_flag'] == 0 else "🔴 WARNING — Over-Voltage Risk!"
    
    return (f"{ir:.6f} Ω", f"{temp_prob:.2f}%", temp_status, volt_status)


def batch_predict(file):
    """Run batch prediction on uploaded CSV."""
    if not MODELS_LOADED:
        return None, None
    if file is None:
        return None, None
    
    df = pd.read_csv(file.name)
    result_df = predict_batch(df, MODELS)
    
    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', prefix='ev_predictions_')
    result_df.to_csv(tmp.name, index=False)
    
    return result_df.head(10), tmp.name


# Build Gradio Interface
with gr.Blocks(
    title="EV Battery Intelligence Predictor",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green")
) as demo:
    
    gr.Markdown("""
    # 🔋 EV Battery Intelligence Predictor
    ### Predict battery aging and safety status using ML models
    
    This application uses trained **Random Forest / XGBoost** models to predict:
    - **Internal Resistance (Ω)** — primary electrochemical aging indicator (R² ≈ 0.97)
    - **Over-Temperature Risk** — probability of thermal anomaly
    - **Over-Voltage Risk** — voltage safety status (rule-based threshold)
    """)
    
    with gr.Tab("🔍 Single Prediction"):
        gr.Markdown("### Enter battery state parameters:")
        
        with gr.Row():
            with gr.Column():
                soc = gr.Slider(0, 100, value=50, step=0.1, label="State of Charge (%)")
                soh = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="State of Health (0-1)")
                terminal_voltage = gr.Slider(2.5, 4.3, value=3.7, step=0.01, label="Terminal Voltage (V)")
                battery_current = gr.Number(value=15.0, label="Battery Current (A)")
                battery_temp = gr.Number(value=35.0, label="Battery Temperature (°C)")
                ambient_temp = gr.Number(value=25.0, label="Ambient Temperature (°C)")
            
            with gr.Column():
                action_current = gr.Number(value=15.0, label="Action Current (A)")
                action_voltage = gr.Number(value=3.7, label="Action Voltage (V)")
                charging_efficiency = gr.Slider(0.7, 1.0, value=0.85, step=0.001, label="Charging Efficiency")
                charging_time = gr.Number(value=2500, label="Charging Time (s)")
        
        predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
        
        with gr.Row():
            ir_output = gr.Textbox(label="Predicted Internal Resistance (Ω)", interactive=False)
            temp_risk = gr.Textbox(label="Over-Temperature Risk", interactive=False)
        with gr.Row():
            temp_status = gr.Textbox(label="Temperature Safety Status", interactive=False)
            volt_status = gr.Textbox(label="Voltage Safety Status", interactive=False)
        
        predict_btn.click(
            fn=single_predict,
            inputs=[soc, soh, terminal_voltage, battery_current, battery_temp,
                    ambient_temp, action_current,
                    action_voltage, charging_efficiency, charging_time],
            outputs=[ir_output, temp_risk, temp_status, volt_status]
        )
    
    with gr.Tab("📊 Batch CSV Prediction"):
        gr.Markdown("""
        ### Upload a CSV file with battery sensor data
        The CSV should contain the same feature columns as the training data.
        Download predictions as a CSV file.
        """)
        
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        batch_btn = gr.Button("🔮 Predict Batch", variant="primary")
        
        preview = gr.Dataframe(label="Predictions Preview (first 10 rows)")
        download = gr.File(label="Download Full Predictions CSV")
        
        batch_btn.click(
            fn=batch_predict,
            inputs=[file_input],
            outputs=[preview, download]
        )

if __name__ == "__main__":
    demo.launch()
