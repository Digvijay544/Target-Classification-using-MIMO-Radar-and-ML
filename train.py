import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide TensorFlow INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from data_generator import generate_data
from preprocess import preprocess_data
from model import build_model

NUM_CLASSES = 5

# Step 1: Load data
X_train, X_test, y_train, y_test = generate_data()

# Step 2: Preprocess data
X_train, y_train = preprocess_data(X_train, y_train, NUM_CLASSES)
X_test, y_test = preprocess_data(X_test, y_test, NUM_CLASSES)

# Step 3: Build model
model = build_model(X_train.shape[1:], NUM_CLASSES)

# Step 4: Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 5: Save model (modern Keras format)
model.save("radar_cnn_model.keras", save_format="keras")
print("Model saved in Keras format")
