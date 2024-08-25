from flask import Blueprint, render_template, request, jsonify
from app.utils.image_processing import load_image_model, predict_image
from app.utils.text_processing import load_text_model, predict_text
import os
import numpy as np
import cv2
routes = Blueprint('routes', __name__)

image_model = load_image_model('app/models/my_model.h5')
text_model = load_text_model('app/models/svm_model.pkl')
# Assume tokenizer and max_length are predefined
tokenizer = ...
max_length = ...

@routes.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image'].read()
            image_array = np.fromstring(image, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_prediction = predict_image(image, image_model)
            return jsonify({'image_prediction': str(image_prediction)})

        if 'text' in request.form:
            text = request.form['text']
            text_prediction = predict_text(text, text_model, tokenizer, max_length)
            return jsonify({'text_prediction': str(text_prediction)})

    return render_template('index.html')
