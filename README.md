# Histopathological Image Classification with Learnable Preprocessing and Custom Architectures

This project was developed as part of our Machine Learning & AI course at BUET (EEE 402, July 2025). It focuses on improving diagnostic accuracy in histopathological images using advanced deep learning techniques.

## 🔍 Project Overview

- **Goal**: Classify colon and lung cancer types using H&E-stained histopathological images.
- **Key Innovations**:
  - Learnable preprocessing using Macenko normalization + U-Net segmentation
  - Optimized augmentation using grid search
  - Custom CNN model (HistPathNet) with CBAM attention
  - Comparative evaluation with ResNet-50, EfficientNet-B0, DenseNet-121, and Vision Transformer

## 🧠 Model Architecture

HistPathNet is a U-Net-inspired architecture enhanced with CBAM attention modules and dual output heads for segmentation and classification.

## 📁 Dataset

- **Name**: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/ambarish/lung-colon-cancer-histopathological-images)
- 25,000 images, 5 classes
- We used a sampled, balanced subset of 2,000 images

## 📊 Results

| Model             | Accuracy | F1 Score | AUROC  |
|------------------|----------|----------|--------|
| ResNet-50        | 96.12%   | 0.9432   | 0.9448 |
| EfficientNet-B0  | 94.80%   | 0.9300   | 0.9320 |
| DenseNet-121     | 94.00%   | 0.9200   | 0.9240 |
| Vision Transformer | 93.00% | 0.9100   | 0.9160 |
| **HistPathNet** (Ours) | **90.00%** | **0.9000** | **0.9600** |
