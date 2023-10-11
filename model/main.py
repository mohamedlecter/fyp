import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image
import json


app = Flask(__name__, template_folder="templates", static_url_path='/static') 

# Load the trained model
model = keras.models.load_model('models/plant_disease_detector')

# Load class_name_lookup from the JSON file
with open('class_name_lookup.json', 'r') as f:
    class_name_lookup = json.load(f)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Create the 'uploads' directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        filename = secure_filename(file.filename)
        image_path = os.path.join('static', filename)
        image_path = 'static/' + filename
        file.save(image_path)


        print("Image path:", filename)

        # Preprocess the image
        img = Image.open(image_path)
        img = img.resize((256, 256))  # Resize to match the model's input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions using the loaded model
        prediction = model.predict(img_array)

        # Get the predicted class index
        # Determine the class index and get the class name from your data
        class_index = np.argmax(prediction)
        class_name = class_name_lookup[str(class_index)]

        # Remove underscores and replace them with spaces
        class_name = class_name.replace("_", " ")

        # Return the prediction in the response
        return render_template('result.html', filename=filename, class_name=class_name)

    return jsonify({'error': 'File format not supported'})





if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
