import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_data(X, y, num_classes):
    X_mag = np.abs(X)
    X_mag = X_mag / np.max(X_mag)
    X_mag = X_mag[..., np.newaxis]
    y_cat = to_categorical(y, num_classes)
    return X_mag, y_cat


def add_gaussian_noise(X, sigma):
    """
    Add Gaussian noise to radar data
    X: input radar cube
    sigma: noise standard deviation
    """
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise
