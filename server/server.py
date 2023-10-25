import os
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import json

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model 2/models/plant_disease_detector')  # Replace 'model_path' with the actual path to your saved model

with open('model 2/class_names.json', 'r') as f:
    class_names = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
        predicted_class = class_names[predicted_class_index]

        # Clean up temporary file
        os.remove(image_path)

        return f'Predicted class: {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)
