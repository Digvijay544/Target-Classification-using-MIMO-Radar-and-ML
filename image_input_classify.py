import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from image_to_radar import image_to_radar
from preprocess import preprocess_data

# ------------------------------
# Configuration
# ------------------------------
NUM_CLASSES = 5

TARGET_NAMES = {
    0: "Cup",
    1: "Bottle",
    2: "Mouse",
    3: "Charger",
    4: "Gum"
}

MODEL_PATH = r"C:\Users\digvi\OneDrive\Desktop\BE Project\paper Implementation\radar_cnn_model.keras"

# ------------------------------
# User input (image)
# ------------------------------
IMAGE_PATH = "image.png"   # user-uploaded image

# ------------------------------
# Convert image → radar sample
# ------------------------------
radar_sample = image_to_radar(IMAGE_PATH)

# ------------------------------
# Preprocess radar sample
# ------------------------------
radar_sample, _ = preprocess_data(
    radar_sample[np.newaxis, ...],
    np.array([0]),
    NUM_CLASSES
)

# ------------------------------
# Load trained radar model
# ------------------------------
model = load_model(MODEL_PATH)

# ------------------------------
# Predict target
# ------------------------------
prediction = model.predict(radar_sample)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

print("\n🎯 TARGET CLASSIFICATION RESULT")
print("--------------------------------")
print("Predicted Target :", TARGET_NAMES[predicted_class])
print(f"Confidence       : {confidence * 100:.2f}%")

# ------------------------------
# Visualize radar as RGB image
# ------------------------------
R = radar_sample[0, :, :, 30, 0]
G = radar_sample[0, :, :, 50, 0]
B = radar_sample[0, :, :, 70, 0]

rgb = np.stack([R, G, B], axis=-1)
rgb = rgb / np.max(rgb)

plt.imshow(rgb)
plt.title(f"Radar RGB Image → Predicted: {TARGET_NAMES[predicted_class]}")
plt.axis("off")
plt.show()
