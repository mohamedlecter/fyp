import os
import numpy as np
from flask import jsonify, request, request_started
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from diagnose.diagnose_model import PlantDiseaseModel
from urllib.parse import urlparse

model = PlantDiseaseModel()


def diagnose_plant():

    if 'image' not in request.files:
        return "No image file provided."

    image = request.files['image']

    if image.filename == '':
        return "No selected file."

    if image:
        # Save the uploaded image to a temporary file
        image_path = 'temp.jpg'
        image.save(image_path)

        # Load and preprocess the input image
        input_image = load_img(image_path, target_size=(224, 224))
        input_image = img_to_array(input_image)
        input_image = preprocess_input(input_image)
        input_image = np.expand_dims(input_image, axis=0)  # Add a batch dimension

        # Make predictions
        predictions = model.predict(input_image)

        # Get the predicted class
        # confidence = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        predicted_class = model.get_class_name(predicted_class_index)

        # Return the result
        result = {
            'disease': predicted_class,
            # 'confidence': confidence
        }

        # Clean up temporary file
        os.remove(image_path)
        return jsonify(result)
