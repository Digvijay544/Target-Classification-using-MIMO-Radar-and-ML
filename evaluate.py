import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_generator import generate_data
from preprocess import preprocess_data

# -----------------------------
# Configuration
# -----------------------------
NUM_CLASSES = 5

TARGET_NAMES = {
    0: "Cup",
    1: "Bottle",
    2: "Mouse",
    3: "Charger",
    4: "Gum"
}

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("radar_cnn_model.keras")
print("✅ Model loaded successfully")

# -----------------------------
# Load and preprocess test data
# -----------------------------
_, X_test, _, y_test = generate_data()
X_test, y_test = preprocess_data(X_test, y_test, NUM_CLASSES)

# -----------------------------
# Predict on test data
# -----------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# -----------------------------
# Show target-wise predictions
# -----------------------------
print("\n🔍 SAMPLE TARGET CLASSIFICATION RESULTS:\n")

for i in range(10):
    actual = TARGET_NAMES[y_true[i]]
    predicted = TARGET_NAMES[y_pred[i]]

    print(f"Sample {i+1}:")
    print(f"  Actual Target    : {actual}")
    print(f"  Predicted Target : {predicted}")
    print("-" * 40)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[TARGET_NAMES[i] for i in range(NUM_CLASSES)]
)

plt.figure(figsize=(7, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Target Classification")
plt.show()

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.mean(y_true == y_pred)
print(f"\n🎯 Overall Classification Accuracy: {accuracy * 100:.2f}%")
