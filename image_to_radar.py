import cv2
import numpy as np

RX, TX, TIME = 20, 20, 100

def image_to_radar(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (TX, RX))

    # Edge detection
    edges = cv2.Canny(img, 50, 150)

    # Feature extraction
    vertical_energy = np.sum(edges, axis=0).mean()
    horizontal_energy = np.sum(edges, axis=1).mean()
    mean_intensity = np.mean(img)

    # Base radar noise
    radar = np.random.randn(RX, TX, TIME) * 0.1

    # Decision logic to generate radar pattern
    if vertical_energy > horizontal_energy * 1.3:
        # Bottle
        radar[:, TX//2-1:TX//2+1, :] += 0.8

    elif horizontal_energy > vertical_energy * 1.3:
        # Mouse
        radar[RX//2-1:RX//2+1, :, :] += 0.8

    elif np.mean(edges) > 25:
        # Charger
        for i in range(RX):
            radar[i, i % TX, :] += 0.7

    elif mean_intensity > 140:
        # Cup
        radar[8:12, 8:12, :] += 0.9

    else:
        # Gum
        for _ in range(15):
            x = np.random.randint(0, RX)
            y = np.random.randint(0, TX)
            radar[x, y, :] += 0.6

    return radar
