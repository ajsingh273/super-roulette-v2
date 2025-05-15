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

@st.cache_resource
def load_lstm_model():
    try:
        return load_model('lstm_next_spin_model.h5')
    except:
        return None

model, scaler = load_nn_model()
lstm_model = load_lstm_model()

# ------------------------------------------
# State Management
# ------------------------------------------
if "live_mode_spins" not in st.session_state:
    st.session_state.live_mode_spins = []

if "live_mode_predictions" not in st.session_state:
    st.session_state.live_mode_predictions = []

if "live_mode_correct" not in st.session_state:
    st.session_state.live_mode_correct = 0

if "live_mode_total" not in st.session_state:
    st.session_state.live_mode_total = 0

# ------------------------------------------
# Prediction Functions
# ------------------------------------------
def predict_two_dozens(spins):
    features = feature_engineering(spins)
    if not features:
        return []

    X = np.array(list(features.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled)[0]
    top_two_indices = np.argsort(probabilities)[-2:][::-1]
    return [i + 1 for i in top_two_indices], probabilities

def predict_next_spin_lstm(spins, seq_len=10):
    if not lstm_model or len(spins) < seq_len:
        return None
    sequence = np.array(spins[-seq_len:]).reshape(1, seq_len)
    prediction = lstm_model.predict(sequence, verbose=0)
    return int(np.argmax(prediction))

def update_accuracy(predicted_dozen, actual_spin):
    if (predicted_dozen == 1 and 1 <= actual_spin <= 12) or \
       (predicted_dozen == 2 and 13 <= actual_spin <= 24) or \
       (predicted_dozen == 3 and 25 <= actual_spin <= 36):
        st.session_state.live_mode_correct += 1
    st.session_state.live_mode_total += 1

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
        spins = [int(x.strip()) for x in history_input.split(',') if x.strip().isdigit()]
        if len(spins) >= 10:
            st.markdown("### 🧠 Prediction")
            top_dozens, probs = predict_two_dozens(spins)
            for d in top_dozens:
                st.markdown(f"<div class='recommendation'><strong>Recommended Dozen:</strong> <span class='highlight'>{d}st dozen</span> (Confidence: {probs[d-1]:.2%})</div>", unsafe_allow_html=True)

            if lstm_model:
                lstm_pred = predict_next_spin_lstm(spins)
                if lstm_pred:
                    st.markdown(f"<div class='recommendation'><strong>LSTM Prediction:</strong> <span class='highlight'>{lstm_pred}</span></div>", unsafe_allow_html=True)

            st.markdown("### 📜 Full History")
            st.code(spins, language="text")
        else:
            st.warning("Please enter at least 10 valid numbers.")
    except Exception as e:
        st.error(f"Error parsing input: {e}")

# ------------------------------------------
# 🔴 Live Mode
# ------------------------------------------
st.subheader("🔴 Live Mode - Real-time Tracking & Prediction")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    new_spin = st.number_input("🎡 New Spin (1–36)", min_value=1, max_value=36, step=1)

with col2:
    if st.button("➕ Add Spin"):
        st.session_state.live_mode_spins.append(new_spin)

        prediction_result = predict_two_dozens(st.session_state.live_mode_spins)
        if prediction_result:
            top_dozens, probs = prediction_result
            st.session_state.live_mode_predictions.append(top_dozens[0])
            update_accuracy(top_dozens[0], new_spin)

with col3:
    st.markdown("#### 🧾 Current Spin History")
    if st.session_state.live_mode_spins:
        st.markdown(f"- **Total Spins Entered:** `{len(st.session_state.live_mode_spins)}`")
        st.markdown(f"- **Last 5 Spins:** `{st.session_state.live_mode_spins[-5:]}`")
    else:
        st.info("No spins entered yet.")

# Show prediction if valid
if len(st.session_state.live_mode_spins) >= 10:
    top_dozens, probs = predict_two_dozens(st.session_state.live_mode_spins)
    st.markdown(f"""
        <div class='recommendation'>
            <strong>Current Dozen Prediction:</strong>
            <span class='highlight'>{top_dozens[0]}st dozen</span>
            (Confidence: {probs[top_dozens[0]-1]:.2%})
        </div>
    """, unsafe_allow_html=True)

    if lstm_model:
        lstm_pred = predict_next_spin_lstm(st.session_state.live_mode_spins)
        if lstm_pred is not None:
            st.markdown(f"""
                <div class='recommendation'>
                    <strong>LSTM Prediction:</strong>
                    <span class='highlight'>{lstm_pred}</span>
                </div>
            """, unsafe_allow_html=True)
else:
    st.warning("At least 10 spins are needed to begin predictions.")

# ------------------------------------------
# Accuracy Metrics
# ------------------------------------------
st.markdown("### 📊 Prediction Accuracy")
if st.session_state.live_mode_total > 0:
    accuracy = (st.session_state.live_mode_correct / st.session_state.live_mode_total) * 100
    st.metric(label="🎯 Accuracy", value=f"{accuracy:.2f}%")
    st.metric(label="✅ Correct Predictions", value=st.session_state.live_mode_correct)
    st.metric(label="📈 Total Predictions", value=st.session_state.live_mode_total)
else:
    st.info("Start adding spins to track prediction accuracy.")
