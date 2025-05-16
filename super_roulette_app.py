import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from ml_model_nn import feature_engineering

# ------------------------------------------
# Setup
# ------------------------------------------
st.set_page_config(page_title="🎯 Super Roulette Assistant", layout="wide")

# Load models and scaler
@st.cache_resource
def load_nn_model():
    model = load_model('roulette_nn_model.h5')
    scaler = joblib.load('scaler_nn.pkl')
    return model, scaler

model, scaler = load_nn_model()

# ------------------------------------------
# Prediction Function: Top Two Dozens
# ------------------------------------------
def predict_two_dozens(spins):
    features = feature_engineering(spins)
    if not features:
        return [], []

    X = np.array(list(features.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled)[0]
    top_two_indices = np.argsort(probabilities)[-2:][::-1]
    return [i + 1 for i in top_two_indices], probabilities

# ------------------------------------------
# Styling
# ------------------------------------------
st.markdown("""
    <style>
        body {
            background-color: #111;
            color: #e0e0e0;
        }
        .stApp {
            background-color: #111;
        }
        h1, h2, .recommendation {
            color: #00ff88;
        }
        .recommendation {
            background: #222;
            border-left: 5px solid #00ff88;
            margin: 10px 0;
            padding: 10px;
        }
        .highlight {
            color: #00ff88;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Title
# ------------------------------------------
st.title("🎯 Super Intelligent Roulette Assistant")
st.subheader("Enter Spin History")

# ------------------------------------------
# History Input Mode
# ------------------------------------------
history_input = st.text_area("📥 Paste previous spin numbers (comma-separated):", height=150)
if history_input:
    try:
        spins = []
        for x in history_input.split(','):
            x = x.strip()
            if x == '00':
                spins.append(37)  # map '00' to 37
            elif x.isdigit():
                spins.append(int(x))

        if len(spins) >= 10:
            st.markdown("### 🧠 Prediction")
            top_dozens, probs = predict_two_dozens(spins)
            for d in top_dozens:
                st.markdown(f"<div class='recommendation'><strong>Recommended Dozen:</strong> <span class='highlight'>{d}st dozen</span> (Confidence: {probs[d-1]:.2%})</div>", unsafe_allow_html=True)

            st.markdown("### 📜 Full History")
            st.code(spins, language="text")
        else:
            st.warning("Please enter at least 10 valid numbers.")
    except Exception as e:
        st.error(f"Error parsing input: {e}")
