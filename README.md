# potatoChipsQualityControl
This project provides a complete pipeline for defect detection in potato slices, designed for food processing applications. The pipeline includes preprocessing potato slice images, training two deep learning models (DenseNet121 and EfficientNetB3) for classification, and using an XGBoost meta-classifier to enhance prediction accuracy through model stacking.

Project Overview
Image Preprocessing: Images of potato slices are resized, adjusted for brightness, and saved with a specified DPI to ensure consistency.

Model Training: Uses two pre-trained models, DenseNet121 and EfficientNetB3, which are fine-tuned on the potato slice dataset.

Meta-Classifier (XGBoost): A meta-model combining predictions from DenseNet121 and EfficientNetB3, aimed at enhancing classification accuracy through ensemble learning.

Features
Image Resizing: Images are resized to a standard 256x256 pixels.
Brightness Adjustment: Brightness is adjusted using Contrast Limited Adaptive Histogram Equalization (CLAHE) for consistency.

DPI Setting: Processed images are saved with a DPI of 300 to maintain quality.

Ensemble Model: Combines DenseNet and EfficientNet predictions through an XGBoost meta-classifier for final classification.

Prerequisites
Python 3.x

Required Libraries:
opencv-python-headless
numpy
pillow
tensorflow
scikit-learn
xgboost
matplotlib (optional, for debugging visualization)
