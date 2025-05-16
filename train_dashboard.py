import streamlit as st
from ml_model_nn import train_model as train_nn_model
from train_lstm import train_lstm_model
import time
import io
import sys

# ------------------------------------------
# Streamlit Setup
# ------------------------------------------
st.set_page_config(page_title="🧠 Model Trainer", layout="centered")
st.title("🎯 Super Roulette Model Trainer")
st.write("Train or retrain your models below:")

# ------------------------------------------
# Utility: Capture stdout to display logs
# ------------------------------------------
class StreamCapture(io.StringIO):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout

# ------------------------------------------
# Sidebar for Model Selection
# ------------------------------------------
st.sidebar.header("🧩 Choose Models to Train")
train_nn = st.sidebar.checkbox("Train Neural Network (Dozen Predictor)", value=True)
train_lstm = st.sidebar.checkbox("Train LSTM (Next-Spin Predictor)", value=False)

# ------------------------------------------
# Train Button
# ------------------------------------------
if st.button("🚀 Start Training"):
    if not train_nn and not train_lstm:
        st.warning("Please select at least one model to train.")
    else:
        with st.expander("📋 Training Log", expanded=True):
            with StreamCapture() as log_output:
                with st.spinner("Training in progress..."):
                    start_time = time.time()
                    if train_nn:
                        print("\n🧠 Training Neural Network (Dozen Predictor)...")
                        train_nn_model()
                    if train_lstm:
                        print("\n🔁 Training LSTM (Next-Spin Predictor)...")
                        train_lstm_model()
                    duration = time.time() - start_time
                    print(f"\n✅ Training complete in {duration:.2f} seconds.")
                st.code(log_output.getvalue(), language="bash")

# ------------------------------------------
# Info
# ------------------------------------------
st.markdown("---")
st.markdown("""
- 🧠 **Neural Network Model:** Learns patterns to predict the most likely dozens based on spin history.
- 🔁 **LSTM Model:** Trains a sequential model to predict the next spin number based on prior sequences.

Models are saved as:
- `roulette_nn_model.h5` (with `scaler_nn.pkl`)
- `lstm_next_spin_model.h5`
""")
