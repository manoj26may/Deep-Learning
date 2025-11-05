# app.py

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import os

# --- 1. CONFIGURATION AND LOADING ASSETS ---
SAVE_DIR = "C:/Users/mmbha/OneDrive/Desktop/Docs/DeepLearning/heart" 
INPUT_FEATURES = ['Age', 'Sex', 'HighBP', 'HighChol', 'Smoker', 'DiffWalk', 'Diabetes', 'GenHlth', 'PhysHlth', 'PhysActivity']

# Load Scaler
SCALER = joblib.load(os.path.join(SAVE_DIR, 'scaler.joblib'))

# Load Models
MODELS = {}
MODEL_FILENAMES = {'ANN': 'ann.keras', 'LSTM': 'lstm.keras', 'RNN': 'rnn.keras'}

for name, filename in MODEL_FILENAMES.items():
    try:
        MODELS[name] = tf.keras.models.load_model(os.path.join(SAVE_DIR, filename))
        print(f"{name} model loaded.")
    except Exception as e:
        print(f"Failed to load {name} model: {e}")


# --- 2. PREDICTION FUNCTION ---

def predict_heart_attack(
    model_choice, Age, Sex, HighBP, HighChol, Smoker, DiffWalk, 
    Diabetes, GenHlth, PhysHlth, PhysActivity
):
    """Processes input, scales it, makes a prediction, and formats the output."""
    
    # 1. Convert inputs to a NumPy array
    raw_data = np.array([[
        Age, Sex, HighBP, HighChol, Smoker, DiffWalk, 
        Diabetes, GenHlth, PhysHlth, PhysActivity
    ]], dtype=np.float32)

    # 2. Scale the data
    scaled_data = SCALER.transform(raw_data)
        
    model = MODELS.get(model_choice)
    if not model:
        return f"Error: Model {model_choice} is not available."

    # 3. Reshape for RNN/LSTM models (Requires 3D input)
    if model_choice in ["LSTM", "RNN"]:
        # [1, 10] -> [1, 1, 10]
        scaled_data = scaled_data.reshape(1, 1, scaled_data.shape[1])
    
    # 4. Predict
    prediction_proba = model.predict(scaled_data, verbose=0).flatten()[0]
    
    # 5. Format results
    prediction_class = "Yes (High Risk)" if prediction_proba > 0.5 else "No (Low Risk)"
    
    return f"Model Used: {model_choice}\n\nProbability of Heart Attack: {prediction_proba:.4f}\nRisk Prediction: {prediction_class}"


# --- 3. GRADIO INTERFACE SETUP ---

# Define the input components for the Gradio interface
inputs = [
    gr.Dropdown(list(MODELS.keys()), label="Select Prediction Model", value="ANN"),
    gr.Slider(minimum=1, maximum=13, step=1, label="Age (1=18-24, 13=80+)", value=""),
    gr.Radio([0, 1], label="Sex (0=Female, 1=Male)", value=""),
    gr.Radio([0, 1], label="HighBP (1=Yes)", value=""),
    gr.Radio([0, 1], label="HighChol (1=Yes)", value=""),
    gr.Radio([0, 1], label="Smoker (1=Yes)", value=""),
    gr.Radio([0, 1], label="DiffWalk (Difficulty Walking, 1=Yes)", value=""),
    gr.Radio([0, 1, 2], label="Diabetes (0=No, 1=Pre, 2=Yes)", value=""),
    gr.Slider(minimum=1, maximum=5, step=1, label="GenHlth (1=Excellent, 5=Poor)", value=""),
    gr.Slider(minimum=0, maximum=30, step=1, label="PhysHlth (Bad Physical Health Days)", value=""),
    gr.Radio([0, 1], label="PhysActivity (1=Yes)", value=""),
]

iface = gr.Interface(
    fn=predict_heart_attack, 
    inputs=inputs, 
    outputs="text",
    title="Multi-Model Heart Attack Risk Predictor",
    description="Select a model and input patient health metrics to get a real-time risk prediction. All models are fine-tuned with class weighting."
)

if __name__ == "__main__":
    iface.launch()