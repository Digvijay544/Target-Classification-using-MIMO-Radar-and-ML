# RadarCNN — Synthetic Radar Target Classification 🚀

**Brief:** A small research / demo project that generates synthetic radar-like 3D data, trains a 3D-CNN to classify five object types (Cup, Bottle, Mouse, Charger, Gum), and includes utilities to evaluate and visualize results and convert a 2D image to a radar-style sample for classification.

---

## 🔧 Features

- Synthetic radar dataset generation (`data_generator.py`) with class-specific patterns
- 3D Convolutional model implemented in `model.py`
- Training script `train.py` that saves a Keras model (`radar_cnn_model.keras`)
- Evaluation and confusion-matrix visualization (`evaluate.py`)
- Noise robustness experiment (`noise_experiment.py`)
- Utility to convert an image to a radar cube (`image_to_radar.py`) and classify with `image_input_classify.py`

---

## ⚙️ Requirements

- OS: Windows (tested)
- Python 3.8+ (virtual environment recommended)
- Key packages: `tensorflow`, `numpy`, `scikit-learn`, `matplotlib`, `opencv-python`

Example (Windows PowerShell):

```powershell
# activate existing venv shipped in project
.\radar_env\Scripts\Activate.ps1

# or create a new venv and install dependencies
python -m venv radar_env
.\radar_env\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow numpy scikit-learn matplotlib opencv-python
```

> Tip: The repository already contains a `radar_env` virtual environment. Activate it before running scripts.

---

## 🚀 Quick Start

1. Activate the environment:

```powershell
.\radar_env\Scripts\Activate.ps1
```

2. Train the model (generates `radar_cnn_model.keras`):

```powershell
python train.py
```

3. Evaluate the saved model:

```powershell
python evaluate.py
```

4. Run the noise experiment:

```powershell
python noise_experiment.py
```

5. Classify a single image (replace `image.png` with your image):

```powershell
python image_input_classify.py
```

---

## 📂 File Overview

- `data_generator.py` — synthetic radar sample generator + train/test split
- `preprocess.py` — magnitude + normalization and one-hot conversion; `add_gaussian_noise`
- `model.py` — 3D CNN architecture and compilation
- `train.py` — end-to-end training pipeline
- `evaluate.py` — loads model, runs predictions and plots confusion matrix
- `noise_experiment.py` — measures accuracy vs Gaussian noise amplitude
- `image_to_radar.py` — naive image → radar conversion used for demo
- `image_input_classify.py` — convert an input image, preprocess, predict and show a visual

---

## 🔧 Configuration Notes

- Global constants (e.g., `RX`, `TX`, `TIME`, `NUM_CLASSES`) are defined across the files and can be changed to alter sample dimensions or class counts.
- The model is saved in modern Keras format at `radar_cnn_model.keras` by `train.py`.
- `preprocess.py` normalizes by `np.max(X_mag)`; if your dataset includes zeros only, watch for divide-by-zero.

---

## 🧪 Reproducibility & Tips

- To get deterministic behavior, set seeds and remove randomness in `data_generator.py` and noise functions.
- If training is slow or runs out of memory, reduce `NUM_SAMPLES` or the model size.
- If using GPU, ensure proper TensorFlow + CUDA/cuDNN versions.

---

## ✨ License & Contact

Use freely for research or demo purposes. If you need help or want improvements (richer synthetic models, dataset export, or more realistic image→radar mapping), open an issue or contact the author.

---

Happy experimenting! ✅
