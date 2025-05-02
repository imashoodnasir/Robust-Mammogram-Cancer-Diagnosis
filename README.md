# Robust-Mammogram-Cancer-Diagnosis

An explainable deep learning framework for binary classification of breast cancer using mammographic images. This repository implements a pipeline using EfficientNetV2 for feature extraction, CBAM for attention refinement, and Grad-CAM++ for visual explanation.

---

## 🧠 Overview

This project addresses two major challenges in breast cancer diagnosis:
1. High accuracy in classification using advanced CNNs.
2. Clinical trust through visual explanations using Grad-CAM++.

The framework was evaluated on three benchmark datasets: **MIAS**, **DDSM**, and **InBreast**.

---

## 📁 Directory Structure

```
Robust-Mammogram-Cancer-Diagnosis/
│
├── preprocess.py        # CLAHE and Z-score normalization
├── augment.py           # Data augmentation layers
├── backbone.py          # EfficientNetV2 feature extractor
├── cbam.py              # Channel and spatial attention (CBAM)
├── classifier.py        # Fully connected classifier head
├── train.py             # Model assembly and training loop
├── gradcampp.py         # Grad-CAM++ visualizations
├── evaluate.py          # Accuracy, precision, recall, F1-score, confusion matrix
└── README.md            # This file
```

---

## 📦 Requirements

Install dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn seaborn
```

---

## 🗂️ Dataset Preparation

Ensure the following:
- All mammograms are in PNG format.
- Images are resized to 224×224.
- Organize images in subfolders:
  ```
  dataset/
  ├── benign/
  └── malignant/
  ```

Use MIAS, DDSM, or InBreast datasets. Preprocess using `preprocess.py`.

---

## 🚀 Usage

### 1. Train the Model
```bash
python train.py
```

### 2. Evaluate Performance
```bash
python evaluate.py
```

### 3. Generate Visual Explanations
```bash
python gradcampp.py
```

---

## 🧪 Model Architecture

- **Backbone**: EfficientNetV2-S (ImageNet pre-trained)
- **Attention**: Convolutional Block Attention Module (CBAM)
- **Classifier**: GAP + Dense layers + Sigmoid
- **Explainability**: Grad-CAM++

---

## 📊 Performance Highlights

| Dataset   | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| MIAS      | 99.85%   | 99.80%    | 99.70% | 99.75%   |
| DDSM      | 99.40%   | 99.30%    | 98.90% | 99.10%   |
| InBreast  | 99.70%   | 99.60%    | 99.50% | 99.55%   |

---

## 🔓 License

This project is licensed under the MIT License.

---

## 🤝 Acknowledgements

- MIAS, DDSM, and InBreast dataset providers.
- TensorFlow and Keras development community.
