import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_data
from preprocess import preprocess_data, add_gaussian_noise
from model import build_model

NUM_CLASSES = 5
NOISE_LEVELS = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
EPOCHS = 5
BATCH_SIZE = 16

accuracies = []

for sigma in NOISE_LEVELS:
    print(f"\nTraining with noise sigma = {sigma}")

    # Load data
    X_train, X_test, y_train, y_test = generate_data()

    # Preprocess FIRST (magnitude + normalization)
    X_train, y_train = preprocess_data(X_train, y_train, NUM_CLASSES)
    X_test, y_test = preprocess_data(X_test, y_test, NUM_CLASSES)

    # Add Gaussian noise AFTER normalization
    X_train = add_gaussian_noise(X_train, sigma)
    X_test = add_gaussian_noise(X_test, sigma)

    # Build model
    model = build_model(X_train.shape[1:], NUM_CLASSES)

    # Train model
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(acc)

    print(f"Accuracy at sigma={sigma}: {acc:.4f}")

# Plot accuracy vs noise
plt.figure(figsize=(7, 5))
plt.plot(NOISE_LEVELS, accuracies, marker='o')
plt.xlabel("Gaussian Noise Standard Deviation (σ)")
plt.ylabel("Classification Accuracy")
plt.title("Accuracy vs Noise Level (RadarCNN)")
plt.grid(True)
plt.show()
