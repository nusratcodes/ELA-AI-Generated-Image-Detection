# Error Level Analysis (ELA) with Custom CNN for AI-Generated Image Detection

**Author:** Nusrat Hossain  
**Date:** 2026

## 📌 Overview
This notebook implements **Error Level Analysis (ELA)** as a preprocessing step combined with a custom **Convolutional Neural Network (CNN)** to classify real and AI-generated images. The model is trained on the **CIFAKE dataset** and evaluated using accuracy, precision, recall, F1-score, and AUC-ROC.

## ⚙️ Pipeline
1. **ELA Extraction:** Re-compress images at JPEG-95 quality, compute pixel-wise difference, and normalize.
2. **Custom CNN:** 4 convolutional blocks enhanced with `BatchNorm` and `MaxPooling`.
3. **Training:** Optimized using the `AdamW` optimizer, `BCE` loss function, across 10 epochs with a batch size of 32.
4. **Evaluation:** Comprehensive performance tracking via Accuracy, AUC, and Confusion Matrix.

## 📊 Dataset Structure
* **Source:** CIFAKE Dataset (resized to 128×128)
* **Training Set:** 2,000 real + 2,000 fake images
* **Validation Set:** 500 real + 500 fake images

## 📈 Results & Evaluation

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.7670 |
| **Precision** | 0.7613 |
| **Recall** | 0.7780 |
| **F1-Score** | 0.7695 |
| **AUC** | 0.8519 |

### 🔍 Key Observation
The model achieves **76.7% accuracy** and **0.85 AUC**, indicating that ELA retains meaningful forensic signals even at 128×128 resolution. Performance is currently limited by the low image size, where JPEG compression artifacts are minimal; utilizing higher-resolution inputs is expected to yield significantly better results.

## 🛠️ Environment & Tools
* **Hardware:** Google Colab (T4 GPU)
* **Frameworks:** PyTorch, OpenCV, scikit-learn
