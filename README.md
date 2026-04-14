Cancer Detection using Deep Learning
Project Overview

This project uses Deep Learning (Convolutional Neural Networks) to detect the presence of brain tumors from MRI images.
It allows users to upload an MRI scan and get a prediction whether a tumor is present or not.

Features:
Upload MRI image
AI-based tumor detection
Fast and accurate prediction
User-friendly web interface (Streamlit)

Technologies Used
1.Python
2.TensorFlow / Keras
3.OpenCV
4.NumPy
5.Streamlit

Model Details
Model Type: Convolutional Neural Network (CNN)
Input Size: 224 × 224 images
Output: Binary classification (Tumor / No Tumor)
Activation Function: ReLU, Sigmoid
Loss Function: Binary Crossentropy

Project Structure
Cancer-Detection/
│
├── app.py                  # Streamlit web app
├── train.py                # Model training script
├── model.py                # CNN model architecture
├── predict.py              # Prediction script
├── data_preprocessing.py   # Data loading & preprocessing
├── cancer_model.h5         # Trained model
└── README.md               # Project documentation

How to Run the Project
1. Install dependencies
pip install tensorflow opencv-python numpy matplotlib streamlit
2. Run the application
streamlit run app.py

Dataset
The model is trained on a Brain MRI dataset containing:

Tumor images
Non-tumor images
Dataset source: Kaggle (Brain Tumor MRI Dataset)

Use Cases
Early detection of brain tumors
Assist doctors in diagnosis
Educational and research purposes

Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
