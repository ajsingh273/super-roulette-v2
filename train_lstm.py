import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------
# Configuration
# ----------------------------------
sequence_length = 10
num_spins = 5000
num_classes = 36  # Predicting numbers 1 to 36

# ----------------------------------
# Generate synthetic roulette spin data
# ----------------------------------
np.random.seed(42)  # For reproducibility
spins = np.random.randint(1, 37, size=num_spins)

# Create sequences and targets
X = np.array([spins[i:i + sequence_length] for i in range(len(spins) - sequence_length)])
y_raw = spins[sequence_length:]  # Next spin after each sequence

# One-hot encode targets (convert to 0-based for categorical)
y = to_categorical(y_raw - 1, num_classes=num_classes)

# ----------------------------------
# Build LSTM model
# ----------------------------------
model = Sequential([
    Embedding(input_dim=37, output_dim=16, input_length=sequence_length),
    LSTM(64),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ----------------------------------
# Train the model
# ----------------------------------
model.fit(
    X, y,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=1
)

# ----------------------------------
# Save the model
# ----------------------------------
model.save('lstm_next_spin_model.h5')
print("✅ LSTM model saved as 'lstm_next_spin_model.h5'")
