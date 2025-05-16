from ml_model_nn import train_model as train_nn_model
from train_lstm import train_lstm_model

import argparse

def main(train_nn: bool, train_lstm: bool):
    if train_nn:
        print("\n=== 🧠 Training Neural Network Dozen Predictor ===")
        train_nn_model()

    if train_lstm:
        print("\n=== 🔁 Training LSTM Next-Spin Predictor ===")
        train_lstm_model()

    if not train_nn and not train_lstm:
        print("⚠️ No models selected to train. Use --nn and/or --lstm flags.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Super Roulette Models")
    parser.add_argument("--nn", action="store_true", help="Train neural network (dozen predictor)")
    parser.add_argument("--lstm", action="store_true", help="Train LSTM (next spin predictor)")

    args = parser.parse_args()
    main(train_nn=args.nn, train_lstm=args.lstm)
