import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

# Define the zones for American Roulette (0 and 00 included)
LEFT_ZONE = {0, 28, 9, 26, 30, 11, 7, 20, 32, 17, 5, 22}
RIGHT_ZONE = {00, 27, 10, 25, 29, 12, 8, 19, 31, 18, 6, 21}
BOTTOM_ZONE = {33, 16, 4, 23, 35, 14, 2, 13, 1, 36, 24, 3, 15, 34}

# -------------------------------
# Feature Engineering with Debugging
# -------------------------------
def feature_engineering(spins):
    if len(spins) < 10:
        return None
    features = {}
    spins_array = np.array(spins)

    # Count the occurrences of each number (1-36)
    for i in range(1, 37):
        features[f'count_{i}'] = np.count_nonzero(spins_array == i)

    # Calculate dozen ratios
    dozen_1 = sum(1 <= s <= 12 for s in spins)
    dozen_2 = sum(13 <= s <= 24 for s in spins)
    dozen_3 = sum(25 <= s <= 36 for s in spins)
    total = len(spins)
    features['dozen_1_ratio'] = dozen_1 / total
    features['dozen_2_ratio'] = dozen_2 / total
    features['dozen_3_ratio'] = dozen_3 / total

    # Calculate momentum for recent spins (10 spins)
    recent = np.array(spins[-10:])
    weights = np.linspace(1.0, 0.1, num=len(recent))
    momentum = [0, 0, 0]
    for idx, s in enumerate(recent):
        if 1 <= s <= 12:
            momentum[0] += weights[idx]
        elif 13 <= s <= 24:
            momentum[1] += weights[idx]
        elif 25 <= s <= 36:
            momentum[2] += weights[idx]
    features['momentum_1'] = momentum[0]
    features['momentum_2'] = momentum[1]
    features['momentum_3'] = momentum[2]

    # Calculate average of last N spins
    features['last_spin'] = spins[-1]
    features['last_2_avg'] = np.mean(spins[-2:])
    features['last_5_avg'] = np.mean(spins[-5:])
    features['last_10_avg'] = np.mean(spins[-10:])

    # Cluster zone features
    features['left_zone_count'] = sum(1 for s in spins if s in LEFT_ZONE)
    features['right_zone_count'] = sum(1 for s in spins if s in RIGHT_ZONE)
    features['bottom_zone_count'] = sum(1 for s in spins if s in BOTTOM_ZONE)

    # Ensuring we return exactly 50 features
    expected_feature_count = 50
    while len(features) < expected_feature_count:
        features[f'additional_feature_{len(features)}'] = 0  # Filling any missing features if necessary

    # Print the number of features generated to debug
    print(f"Number of features generated: {len(features)}")
    return features

# -------------------------------
# Train Neural Network Model
# -------------------------------
def train_model():
    print("Training NN model...")

    X_data, y_data = [], []

    for _ in range(5000):
        spins = np.random.choice(range(1, 37), size=50, replace=True).tolist()
        features = feature_engineering(spins)
        if features:
            X_data.append(list(features.values()))
            next_spin = np.random.randint(1, 37)
            if 1 <= next_spin <= 12:
                y_data.append(0)
            elif 13 <= next_spin <= 24:
                y_data.append(1)
            else:
                y_data.append(2)

    feature_names = list(feature_engineering(list(range(1, 51))).keys())
    df = pd.DataFrame(X_data, columns=feature_names)
    labels = pd.Series(y_data)

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=1)

    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    print(classification_report(y_test, y_pred))

    model.save('roulette_nn_model.h5')
    joblib.dump(scaler, 'scaler_nn.pkl')

    print("✅ NN model and scaler saved.")

# -------------------------------
# Prediction Function
# -------------------------------
def predict_next_spin(spins):
    features = feature_engineering(spins)
    if features is None:
        return "Not enough data to predict"
    
    # Ensure that the features are the same as those used in training
    print(f"Number of features in prediction: {len(features)}")
    
    # Preprocess the features just like we did during training
    features_array = np.array(list(features.values())).reshape(1, -1)
    features_scaled = scaler.transform(features_array)  # Use the trained scaler
    
    # Predict using the trained model
    prediction = model.predict(features_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if predicted_class == 0:
        return "Left Zone"
    elif predicted_class == 1:
        return "Right Zone"
    else:
        return "Bottom Zone"
