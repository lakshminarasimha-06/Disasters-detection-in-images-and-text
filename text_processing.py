import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib

def load_text_model(model_path):
     return joblib.load(model_path)

def preprocess_text(text, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences

def predict_text(text, model, tokenizer, max_length):
    preprocessed_text = preprocess_text(text, tokenizer, max_length)
    predictions = model.predict(preprocessed_text)
    return np.argmax(predictions, axis=1)
