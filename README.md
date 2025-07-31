# ADVANCING-HISTOPATHOLOGICAL-IMAGE-CLASSIFICATION-THROUGH-LEARNABLE-PREPROCESSING

# Cancer Detection using Deep Learning

This project applies U-Net and ResNet architectures on histopathological images to classify lung and colon cancer types.

## Features
- Macenko stain normalization
- U-Net nuclei segmentation
- ResNet-50, EfficientNet-B0 etc comparisons
- Custom HistPathNet model

## Getting Started
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run training: `python src/train.py`

## Results
Accuracy: 96.12% | F1-score: 0.9432 | AUROC: 0.9448
