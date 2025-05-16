import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Constants for zone categorization (example)
LEFT_ZONE = list(range(1, 13))
RIGHT_ZONE = list(range(13, 25))
BOTTOM_ZONE = list(range(25, 37))

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
    current_feature_count = len(features)

    # If the number of features is less than expected, add additional features
    while len(features) < expected_feature_count:
        features[f'additional_feature_{len(features)}'] = 0  # Filling any missing features if necessary

    # Print the number of features generated to debug
    print(f"Number of features generated: {len(features)}")
    return features

def train_model(spins):
    X = []
    y = []
    
    # Generate features and labels
    for i in range(10, len(spins)):
        features = feature_engineering(spins[i-10:i])
        if features:
            X.append(list(features.values()))
            y.append(spins[i])  # Target is the next spin

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape data for LSTM (samples, time steps, features)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model and scaler
    model.save('lstm_next_spin_model.h5')
    with open('scaler_nn.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def predict_next_spins(spins):
    # Load the trained model and scaler
    with open('scaler_nn.pkl', 'rb') as model_file:
        scaler = pickle.load(model_file)
    
    model = tf.keras.models.load_model('lstm_next_spin_model.h5')

    # Prepare the input features
    features = feature_engineering(spins)
    if not features:
        return None

    # Scale the features
    features_scaled = scaler.transform([list(features.values())])

    # Reshape input to match LSTM expected input
    features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

    # Predict the next spin
    prediction = model.predict(features_scaled)
    return prediction[0][0]

# Sample usage
if __name__ == "__main__":
    # Example spins data
    spins_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 1, 2, 3, 4, 5]

    # Train the model
    train_model(spins_data)

    # Predict the next spin
    next_spin_prediction = predict_next_spins(spins_data)
    print(f"Predicted next spin: {next_spin_prediction}")
