# app/api/diagnose/controllers/diagnose_controller.py
import os
import numpy as np
from flask import request
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from diagnose.diagnose_model import PlantDiseaseModel

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
        predicted_class_index = np.argmax(predictions)
        predicted_class = model.get_class_name(predicted_class_index)

        # Clean up temporary file
        os.remove(image_path)

        return f'Predicted class: {predicted_class}'
