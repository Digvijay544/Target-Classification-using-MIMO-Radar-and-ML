import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hides TF INFO/WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.model_selection import train_test_split

RX = 20
TX = 20
TIME_SAMPLES = 100
NUM_CLASSES = 5
NUM_SAMPLES = 500

def create_pattern(label):
    """Create class-specific radar patterns"""
    data = np.random.randn(RX, TX, TIME_SAMPLES) * 0.1

    if label == 0:  # Cup → center strong reflection
        data[8:12, 8:12, :] += 3

    elif label == 1:  # Bottle → vertical reflection
        data[:, 10:12, :] += 2

    elif label == 2:  # Mouse → horizontal reflection
        data[10:12, :, :] += 2

    elif label == 3:  # Charger → diagonal reflection
        for i in range(RX):
            data[i, i % TX, :] += 2

    elif label == 4:  # Gum → sparse reflections
        for _ in range(20):
            x = np.random.randint(0, RX)
            y = np.random.randint(0, TX)
            data[x, y, :] += 2

    return data

def generate_data():
    X = []
    y = []

    for _ in range(NUM_SAMPLES):
        label = np.random.randint(0, NUM_CLASSES)
        sample = create_pattern(label)
        X.append(sample)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)
