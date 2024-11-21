import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


IMAGE_HEIGHT = 232
IMAGE_WIDTH = 232



def preprocessing_clahe(image):
    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)    
    image = clahe.apply(image)
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    return image


def predict_single_image(image_arr, model):

    # Apply CLAHE
    image = preprocessing_clahe(image_arr)
    
    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image)

    # Calculate probabilities
    prediction_probability = prediction[0][0]  # Assuming binary classification output
    predicted_class_name = 'TB Detected' if prediction_probability > 0.5 else 'TB Not Detected'
    prediction_class = prediction_probability if prediction_probability >= 0.5 else (1-prediction_probability)

    return predicted_class_name, prediction_class
