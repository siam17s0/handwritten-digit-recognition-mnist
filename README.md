# handwritten-digit-recognition-mnist

Live Demo: https://mnist-digit-recognition-cnn.streamlit.app


# Project Title
Handwritten Digit Recognition using CNN vs RNN

# Project Overview
This project compares the performance of CNN and RNN models on the MNIST dataset for handwritten digit classification. 
The goal is to analyze which architecture performs better for image-based digit recognition.

# Dataset
Dataset: MNIST
Total images: 70,000
Training: 60,000
Testing: 10,000
Image size: 28 × 28 grayscale
Classes: 0–9 digits

# Models Used
1.Convolutional Neural Network (CNN)
2.Recurrent Neural Network (RNN)

## Model explaination
CNN:
1.Better for spatial feature extraction
2.Convolution + pooling layers

RNN:
1.Image treated as sequence
2.Captures sequential patterns

# Model Architecture

## Example CNN:
Input (28x28)
↓
Conv2D
↓
MaxPooling
↓
Conv2D
↓
Flatten
↓
Dense
↓
Softmax

## Example RNN:
Input (28x28)
↓
Reshape (sequence)
↓
LSTM / SimpleRNN
↓
Dense
↓
Softmax

# Results
Model	  	Test Accuracy
CNN	          99.29
RNN           8.92




