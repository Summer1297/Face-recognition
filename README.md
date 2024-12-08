# Face Recognition Deep Learning Model

A Convolutional Neural Network (CNN) based face recognition model trained on the LFW (Labeled Faces in the Wild) dataset.

## Project Overview

This model uses deep learning techniques for facial image classification. Key features include:

- Uses LFW dataset (minimum 20 images per person)
- CNN architecture for feature extraction and classification
- Data augmentation for improved generalization
- Dynamic learning rate adjustment during training

## Dependencies

- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Keras

## Model Architecture

The model consists of:
- 3 Convolutional blocks (Conv2D + ReLU + MaxPooling2D)
- Flatten layer
- Fully connected layer (512 neurons)
- Dropout layer (0.5)
- Softmax output layer

## Training Details

- Dataset split: 80% training, 20% testing
- Training epochs: 30
- Batch size: 64
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy

## Model Performance

- Test accuracy: ~80%
- ROC-AUC score: 0.99
