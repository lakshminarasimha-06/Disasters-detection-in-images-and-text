import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def load_image_model(model_path):
    return load_model(model_path)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image, model):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    return np.argmax(predictions, axis=1)
