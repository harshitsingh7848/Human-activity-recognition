# Human Activity Recognition on Edge Devices

CNN-LSTM and GRU classifiers trained on UCI HAR and WISDM, compressed via PTQ, QAT, and structured pruning, and deployed on Android (Pixel 6) using TensorFlow Lite — achieving **93% accuracy at 7.8 ms latency** from a **0.22 MB** model.

---

## Overview

This project builds an end-to-end mobile HAR system using raw inertial signals from smartphones. Two compact recurrent architectures (CNN-LSTM and GRU) are trained, compressed through a systematic optimization pipeline, and deployed on Android via TFLite. An autoencoder-based anomaly rejection module filters out unreliable sensor windows before reporting predictions.

**Key result:** CNN-LSTM compressed from **0.84 MB to 0.22 MB** via PTQ Dynamic, maintaining **85.85% accuracy** on UCI HAR at **7.8 ms latency** on a Pixel 6. Dense-only QAT variant achieves **92.97% accuracy** at 0.83 MB.

---

## Datasets

| Dataset | Sensors | Sampling Rate | Activities | Notes |
|---|---|---|---|---|
| UCI HAR | Acc + Gyro (6 channels) | 50 Hz | 6 | Fixed windows, T = 128 |
| WISDM | Acc only (Gyro zero-padded) | ~20 Hz | 6 | Raw series segmented into windows |

Both datasets are segmented into fixed-length windows of shape (128, 6). WISDM gyroscope channels are zero-padded to maintain architectural consistency with UCI HAR.

---

## Model Architectures

### CNN-LSTM (218,054 params)
- Two Conv1D layers: 64 filters kernel size 5, then 128 filters kernel size 3, each with BatchNorm and ReLU
- Two stacked LSTM layers: 128 units (return sequences), then 64 units
- Dense head: Dense(128) + BatchNorm + Dropout + Dense(num classes) with softmax

### GRU (28,742 params)
- GRU(64, return sequences=True)
- GRU(32, return sequences=True)
- Flatten + Dense(128) + BatchNorm + Dropout + softmax

GRU is significantly more compact than CNN-LSTM and is the recommended architecture for tight memory budgets.

---

## Compression Pipeline

| Stage | Technique | Description |
|---|---|---|
| Baseline | Float32 | Full-precision starting point |
| PTQ Dynamic | Post-Training Quantization | Weights quantized to int8, activations remain float at runtime |
| PTQ Int8 | Full Integer Quantization | Both weights and activations quantized to int8 using a representative dataset |
| QAT | Dense-only QAT | QAT applied to Dense layers only due to Conv1D and RNN layer-support constraints in TFLite |
| Pruning | Magnitude-based Structured Pruning | Polynomial decay schedule, gradually increasing sparsity |

**Important toolchain constraints encountered:**
- INT16x8 mixed-precision formats are not stable for RNN-heavy models; INT8 was used throughout
- QAT does not support Conv1D or recurrent layers in this setup, so only the Dense head was quantized
- `tf.TensorListReserve` requires a static `element_shape` during TFLite conversion; resolved by disabling experimental tensor list lowering
- `SELECT_TF_OPS` caused runtime failures on Android; resolved by exporting with a concrete function signature for static shape specification

---

## Results

### UCI HAR

| Architecture | Variant | Test Accuracy | TFLite Size (MB) |
|---|---|---|---|
| CNN-LSTM | Float32 | 86.32% | 0.84 |
| CNN-LSTM | PTQ Dynamic | 85.85% | 0.22 |
| CNN-LSTM | PTQ Int8 | 84.86% | 0.224 |
| CNN-LSTM | Dense-only QAT | 92.97% | 0.83 |
| CNN-LSTM | Pruned | 82.87% | 1.67 |
| GRU | Float32 | 91.65% | 0.155 |
| GRU | PTQ Dynamic | 91.68% | 0.078 |
| GRU | PTQ Int8 | 76.42% | 0.078 |
| GRU | Dense-only QAT | 85.34% | 0.387 |
| GRU | Pruned | 90.43% | 0.270 |

### WISDM

| Architecture | Variant | Test Accuracy | TFLite Size (MB) |
|---|---|---|---|
| CNN-LSTM | Float32 | 83.20% | 0.83 |
| CNN-LSTM | PTQ Dynamic | 83.12% | 0.22 |
| CNN-LSTM | PTQ Int8 | 82.92% | 0.22 |
| CNN-LSTM | Dense-only QAT | 82.75% | 0.83 |
| GRU | Float32 | 84.11% | 0.155 |
| GRU | PTQ Dynamic | 84.62% | 0.078 |
| GRU | PTQ Int8 | 82.86% | 0.078 |
| GRU | Dense-only QAT | 84.68% | 0.83 |

### UCI HAR Per-Class Performance (CNN-LSTM Float32, Overall Accuracy: 93%)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| WALKING | 0.98 | 0.97 | 0.97 |
| WALKING UPSTAIRS | 0.99 | 0.97 | 0.98 |
| WALKING DOWNSTAIRS | 0.96 | 0.99 | 0.98 |
| SITTING | 0.85 | 0.91 | 0.88 |
| STANDING | 0.89 | 0.95 | 0.92 |
| LAYING | 0.95 | 0.82 | 0.88 |

Dynamic activities (walking, stairs) achieve F1 above 0.97. Postural activities (sitting, laying) are harder to distinguish due to similar inertial signatures.

---

## Anomaly Rejection Module

An autoencoder is trained on normal activity windows. At inference time, windows with reconstruction error above a threshold (98th percentile of validation set) are flagged as anomalous and skipped, reducing erroneous predictions under ambiguous sensing conditions.

Anomaly score: mean squared reconstruction error over the window tensor of shape (T, C).

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

Place UCI HAR and WISDM raw datasets under `data/` before running notebooks.

---

## How to Run

All notebooks are under `notebooks/`.

### Step 1: Data Preprocessing

Load and window UCI HAR and WISDM raw inertial signals into tensors of shape (N, 128, 6).

### Step 2: Model Training

Train CNN-LSTM and GRU classifiers. Save float32 models to `models/`.

### Step 3: Compression

Apply PTQ (dynamic and int8), dense-only QAT, and structured pruning. Convert to `.tflite`.

### Step 4: Anomaly Module

Train the autoencoder on normal windows. Set rejection threshold at 98th percentile of reconstruction error on a validation set.

### Step 5: Android Deployment

Load `.tflite` models in the Android app via TFLite interpreter. Profile latency and CPU usage using Android Studio Profiler (Trepn Profiler is not compatible with Pixel 6).

---

## Repository Structure

```
.
├── notebooks/       # Training, compression, anomaly, and evaluation notebooks
├── src/             # Data loading, windowing, model definitions, compression utilities
├── models/          # Saved float32 .h5 and compressed .tflite models
├── data/            # UCI HAR and WISDM datasets (download separately)
└── figures/         # Confusion matrices, training curves, accuracy-size plots
```

---

## Key Takeaways

- PTQ Dynamic is the best compression choice for CNN-LSTM: 74% size reduction with less than 1% accuracy drop on UCI HAR.
- GRU + PTQ Dynamic achieves 91.68% accuracy at just 0.078 MB, the most efficient configuration overall.
- QAT layer-support limitations in TFLite mean only the Dense head can be quantized for RNN architectures, limiting file-size gains from QAT.
- Pruning increases file size in this setup due to sparse weight storage overhead; runtime sparsity support would be needed for real latency gains.
- Concrete function signatures are required for stable TFLite export of recurrent models; standard Keras export may introduce 2-3 point accuracy shifts.

---

## Authors

**Harshit Singh**, Colorado State University, harshit.singh@colostate.edu

**Trisha Ghali**, Colorado State University, trisha.ghali@colostate.edu

---

## References

Full references available in the project report. Key works include Milenkoski et al. (MIPRO 2018), Zebin et al. (IEEE BHI 2019), Nweke et al. (Expert Systems with Applications, 2018), and Xia et al. (IEEE Access, 2020).
