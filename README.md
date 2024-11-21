# TB Detection
Hosted App - [link](https://Tb-detection.streamlit.app)

## Overview
This is an Omdena project for the Pune India local chapter - Early Detection of Tuberculosis.

This repository contains a deep learning model built using TensorFlow and Keras to classify chest X-ray images as either "TB Positive" or "TB Negative." The model utilizes a pre-trained **ResNet50** architecture for feature extraction and is fine-tuned on a custom dataset. Additionally, a web application built with **Streamlit** is included, allowing users to upload X-ray images and receive predictions on whether the image shows tuberculosis.

## Dataset
The dataset consists of:
- **TB Images**: 800 images labeled as positive for tuberculosis.
- **Non-TB Images**: 3800 images labeled as negative for tuberculosis.

### Data Splitting
The dataset is divided into:
- **Training Set**: 70% of the data
- **Validation Set**: 20% of the data
- **Test Set**: 10% of the data

### Directory Structure
The data is organized into the following directory structure:


```
basedir/
├── train/
│   ├── tb/
│   └── not_tb/
├── validation/
│   ├── tb/
│   └── not_tb/
└── test/
    ├── tb/
    └── not_tb/
```



## Model Architecture
The model uses the ResNet50 architecture without the top fully connected layers, allowing for custom classification layers to be added. The architecture consists of:
- Global Average Pooling layer
- Dense layer with 512 neurons and ReLU activation
- Dropout layer for regularization
- Output layer with a sigmoid activation function for binary classification

## Preprocessing
Images are preprocessed using:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for contrast enhancement.
- Data augmentation techniques, including:
  - Horizontal flips
  - Random rotations
  - Brightness adjustments
  - Zooming

## Training
The model is trained using:
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam with a learning rate of 0.001
- **Metrics**: Accuracy, Recall, Precision

Early stopping and learning rate reduction techniques are employed to enhance training efficiency.

## Evaluation
The model is evaluated on the test set, providing:
- **Accuracy**: ~99%
- **Precision**: 100% for TB class
- **Recall**: ~96% for TB class
- **F1 Score**: ~0.9787
- **AUC Score**: ~0.9989

### Confusion Matrix
A confusion matrix is generated to visualize the model's performance, indicating true positive, true negative, false positive, and false negative predictions.

## Model Saving
The trained model is saved as `resnet_tuned_model.h5` for future inference and deployment.

## Usage
To use the model for predictions, load the saved model and pass the preprocessed X-ray images to the model's `predict` method.

```
from tensorflow.keras.models import load_model

model = load_model('resnet_tuned_model.h5')
predictions = model.predict(preprocessed_images)
```

## Model Architecture
The model uses the **ResNet50** architecture without the top fully connected layers, allowing for custom classification layers to be added. The architecture consists of:
- **Global Average Pooling** layer
- **Dense** layer with 512 neurons and ReLU activation
- **Dropout** layer for regularization
- **Output** layer with a sigmoid activation function for binary classification

## Preprocessing
Images are preprocessed using:
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for contrast enhancement.
- Data augmentation techniques, including:
  - Horizontal flips
  - Random rotations
  - Brightness adjustments
  - Zooming

## Training
The model is trained using:
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam with a learning rate of 0.001
- **Metrics**: Accuracy, Recall, Precision

Early stopping and learning rate reduction techniques are employed to enhance training efficiency.

## Evaluation
The model is evaluated on the test set, providing:
- **Accuracy**: ~99%
- **Precision**: 100% for TB class
- **Recall**: ~96% for TB class
- **F1 Score**: ~0.9787
- **AUC Score**: ~0.9989

### Confusion Matrix
A confusion matrix is generated to visualize the model's performance, indicating true positive, true negative, false positive, and false negative predictions.

## Model Saving
The trained model is saved as `resnet_tuned_model.h5` for future inference and deployment.

## Usage
To use the model for predictions, load the saved model and pass the preprocessed X-ray images to the model's `predict` method:

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('resnet_tuned_model.h5')

# Make predictions on preprocessed images
predictions = model.predict(preprocessed_images)


## Conclusion
This model serves as a robust solution for TB detection from X-ray images, leveraging deep learning techniques to improve accuracy and reliability in medical diagnostics.
