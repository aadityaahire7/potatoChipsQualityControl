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
matplotlib

Steps to Run the Project
Clone the Repository: Download the project to your local machine.


git clone <repository-url>
cd <project-directory>


pip install opencv-python-headless numpy pillow tensorflow scikit-learn xgboost matplotlib

Prepare Dataset: Ensure your dataset is organized as follows:


dataset/
├── Processed Defected Slices/  # Images of defected potato slices
└── Processed Potato Slices/    # Images of healthy potato slices

Run the Image Preprocessing Script: This script will resize, adjust brightness, and save images in the Processed Potato slice/ folder with a DPI of 300.

python preprocess_images.py
Run the Training Script: Train the DenseNet121 and EfficientNetB3 models, followed by the XGBoost meta-classifier.


python train_model.py

Evaluate the Model: Check the accuracy and confusion matrix after training to assess model performance.
