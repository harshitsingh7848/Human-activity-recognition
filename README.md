# Human Activity Recognition on Edge Devices

CNN-LSTM and GRU classifiers trained on UCI HAR and WISDM, compressed via PTQ, QAT, and structured pruning, and deployed on Android (Pixel 6) using TensorFlow Lite, achieving **93% accuracy at 7.8 ms latency** from a **0.22 MB** model.

---

## Overview

This project explores end-to-end compression and edge deployment of deep learning models for human activity recognition. Starting from full-precision CNN-LSTM and GRU architectures, we apply a progressive compression pipeline (PTQ → QAT → Pruning) and evaluate the accuracy-latency-size tradeoff across optimization strategies. The final compressed model runs entirely on-device with no server dependency.

**Key result:** Model size reduced from **0.84 MB → 0.22 MB** while maintaining **93% accuracy** at **7.8 ms inference latency** on a Pixel 6.

---

## Datasets

| Dataset | Activities | Sensors | Samples |
|---|---|---|---|
| [UCI HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 (walking, sitting, standing, etc.) | Accelerometer + Gyroscope | 10,299 |
| [WISDM](https://www.cis.fordham.edu/wisdm/dataset.php) | 6 (walking, jogging, upstairs, etc.) | Accelerometer | 1,098,207 |

---

## Model Architecture

Two sequence classifiers were trained and compared:

- **CNN-LSTM** : Convolutional layers for local feature extraction, followed by LSTM for temporal modeling
- **GRU** : Gated Recurrent Unit as a lightweight alternative to LSTM for on-device inference

---

## Compression Pipeline

| Stage | Technique | Description |
|---|---|---|
| Baseline | Full-precision float32 | Starting point : 0.84 MB |
| PTQ | Post-Training Quantization | int8 weights, no retraining required |
| QAT | Quantization-Aware Training | Simulates quantization during fine-tuning for better accuracy |
| Pruning | Structured Pruning | Removes entire filters/channels; reduces parameter count |
| Final | Combined | 0.22 MB model at 93% accuracy, 7.8 ms latency on Pixel 6 |

---

## Setup

### 1. Install dependencies

```bash
pip install tensorflow tensorflow-model-optimization numpy pandas scikit-learn matplotlib
```

### 2. Clone the repo

```bash
git clone https://github.com/harshitsingh7848/Human-activity-recognition.git
cd Human-activity-recognition
```

### 3. Download datasets

Place UCI HAR and WISDM datasets under `data/` before running notebooks.

---

## How to Run

All notebooks are under `notebooks/`. Run them in order:

### Step 1 : Data Preprocessing

```
notebooks/01_data_preprocessing.ipynb
```
Loads UCI HAR and WISDM, normalizes sensor signals, segments into sliding windows.

### Step 2 : Model Training

```
notebooks/02_model_training.ipynb
```
Trains CNN-LSTM and GRU classifiers. Saves full-precision `.h5` models to `models/`.

### Step 3 : Compression

```
notebooks/03_compression.ipynb
```
Applies PTQ, QAT, and structured pruning. Converts to `.tflite` format for Android deployment.

### Step 4 : Evaluation

```
notebooks/04_evaluation.ipynb
```
Benchmarks accuracy, model size, and latency across all compression strategies. Generates comparison plots under `figures/`.

### Android Deployment

TFLite models are located under `models/tflite/`. Load with the TensorFlow Lite Android library for on-device inference.

---

## Results

| Model | Size | Accuracy | Latency (Pixel 6) |
|---|---|---|---|
| CNN-LSTM (baseline) | 0.84 MB | ~95% | ~18 ms |
| CNN-LSTM + PTQ | ~0.28 MB | ~94% | ~10 ms |
| CNN-LSTM + QAT | ~0.25 MB | ~94% | ~9 ms |
| CNN-LSTM + Pruning + QAT | **0.22 MB** | **93%** | **7.8 ms** |

---

## Repository Structure

```
.
├── notebooks/          # Step-by-step Jupyter notebooks
├── src/                # Preprocessing, model definitions, compression utilities
├── models/             # Saved .h5 and .tflite models
├── data/               # UCI HAR and WISDM datasets (not included, download separately)
└── figures/            # Accuracy/latency/size comparison plots
```

---

## Key Takeaways

- QAT consistently outperforms PTQ in post-compression accuracy, at the cost of additional training time.
- Structured pruning is essential for real latency gains on mobile — unstructured sparsity alone doesn't translate to speedup on edge hardware.
- The compressed model remains reliable for health monitoring wearables without sacrificing meaningful accuracy.

---

## Author

**Harshit Singh**
- GitHub: [harshitsingh7848](https://github.com/harshitsingh7848)
- LinkedIn: [harshit-singh96](https://linkedin.com/in/harshit-singh96)
