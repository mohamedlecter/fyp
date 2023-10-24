import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image
import json


app = Flask(__name__, template_folder="templates", static_url_path='/static') 

# Load the trained model
model = keras.models.load_model('model/models/plant_disease_detector')

# Load class_name_lookup from the JSON file
with open('model/class_name_lookup.json', 'r') as f:
    class_name_lookup = json.load(f)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a function to preprocess the image

def preprocess_image(image):
    # Resize the image to the input size expected by the model (256x256)
    img = image.resize((256, 256))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        image_file = request.files['image']
        if image_file:
            image = Image.open(image_file)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Make predictions using your plant disease detection model
            predictions = model.predict(preprocessed_image)

            # Process the predictions as needed
            class_index = np.argmax(predictions)
            class_name = class_name_lookup[str(class_index)]
            class_name = class_name.replace('_', ' ')

            return f"Predicted class: {class_name}"
        else:
            return "No image file found in the request", 400
    except Exception as e:
        return str(e), 500



if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
