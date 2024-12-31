# Birds Classification Using Voice Data

## Overview
This project is a deep learning-based solution for classifying bird species using their vocalizations. By analyzing bird call recordings, the model identifies the bird species with high accuracy. This project leverages spectrogram analysis and state-of-the-art neural networks to process and classify the audio data.

## Features
- Converts bird audio recordings into spectrograms for analysis.
- Employs deep learning techniques such as Convolutional Neural Networks (CNNs) for classification.
- Includes pre-processing steps to clean and normalize audio data.
- Provides visualization tools for spectrograms and classification results.
- Supports the addition of new bird species for continuous improvement.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** 
  - NumPy
  - Pandas
  - Matplotlib
  - Librosa
  - TensorFlow/Keras
- **Tools:** Jupyter Notebook, VSCode
- **Deep Learning Framework:** TensorFlow/Keras

## Dataset
The dataset used consists of bird call audio recordings sourced from publicly available repositories such as [Xeno-Canto](https://www.xeno-canto.org/) or other similar platforms. Each recording is labeled with the corresponding bird species.

### Preprocessing
- **Audio Cleaning:** Removes background noise and trims silence.
- **Feature Extraction:** Converts audio to Mel-spectrograms for deep learning input.
- **Normalization:** Scales spectrogram values for better model performance.

## Model Architecture
- The model is a Convolutional Neural Network (CNN) tailored for spectrogram analysis.
- Key layers include:
  - Convolutional layers for feature extraction.
  - Pooling layers for dimensionality reduction.
  - Dense layers for classification.

## Results
- Achieved an accuracy of **84%** on the test dataset.
- Visualized spectrograms and model predictions for validation.

