# Project-3
Developed an deep learning-based model to classify chest X-ray images as either normal or showing signs of tuberculosis (TB). The system will preprocess and augment image data, train multiple deep learning models, and evaluate their performance.


 Tuberculosis Detection from Chest X-Ray Images using Deep Learning
Problem Statement
This project aims to develop a deep learning-based system to classify chest X-ray images as normal or indicating tuberculosis (TB). The system includes data preprocessing, model training and evaluation, and a Streamlit interface for real-time image classification.

Business Use Cases
Early Detection of Tuberculosis
Assist radiologists and healthcare professionals by providing fast and accurate TB diagnosis.

Automated Screening in Remote Areas
Enable TB detection where access to radiologists is limited or unavailable.

Reducing Diagnostic Errors
Serve as an AI-powered second opinion to support clinical decisions.

Research and Analysis
Facilitate analysis of TB patterns and model performance for future improvements.

Project Approach
1. Data Preparation
Dataset: tuberculosis-chest-x-rays-images

Total Images: 3008 (TB: 2494, Normal: 514)

Split into training, validation, and test sets.

2. Preprocessing & Augmentation
Resized images for uniformity.

Normalized pixel values.

Applied data augmentation (rotation, flipping, zoom).

Ensured class balance and handled missing/corrupt data.

3. Exploratory Data Analysis (EDA)
Visualized class distributions and pixel intensity histograms.

4. Model Development
Transfer Learning with:

ResNet50

VGG16

EfficientNetB0

Tuned hyperparameters and regularization techniques.

5. Evaluation Metrics
Accuracy

Precision

Recall

F1-score

ROC-AUC Curve

6. Application Development
Streamlit-based interface for image upload and real-time TB prediction.

Accepts chest X-ray images and outputs prediction results with confidence scores.
