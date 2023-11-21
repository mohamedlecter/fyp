import os
import pymongo
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import json
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from user.models import User, bcrypt  
from pymongo import MongoClient


app = Flask(__name__)


CONNECTION_STRING =  'mongodb+srv://admin:admin@fyp.sabqf4f.mongodb.net/?retryWrites=true&w=majority'
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('fyp')
user_collection = pymongo.collection.Collection(db, 'user')

bcrypt = Bcrypt(app)


@app.route("/test")
def test():
    db.user_collection.insert_one({"name": "mohamed"})
    return "Connected to the data base!"

@app.route('/user/signup', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')
    email = request.json.get('email')

    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    existing_user = db.db.users.find_one({'username': username})
    if existing_user:
        return jsonify({'error': 'Username already exists'}), 400

    new_user = User(username, password, email)
    db.db.users.insert_one({'username': new_user.username, 'password': new_user.password, 'email': new_user.email})

    return jsonify({'message': 'Registration successful'}), 201

@app.route('/user/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')
    
    if not email or not password:
        return jsonify({'error': 'email and password are required'}), 400

    user = db.db.users.find_one({'email': email})

    if user and bcrypt.check_password_hash(user['password'], password):
        # You can implement token-based authentication here
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401
    
@app.route('/user/', methods=['GET'])
def get_all_users():
    users = db.db.users.find()
    user_list = []
    for user in users:
        user_list.append({
            'username': user['username'],
            'email': user['email']
        })
    return jsonify(user_list), 200



# Load the trained model
model = tf.keras.models.load_model('../model 2/models/plant_disease_detector')  # Replace 'model_path' with the actual path to your saved model

with open('../model 2/class_names.json', 'r') as f:
    class_names = json.load(f)

@app.route('/', methods=['GET'])
def index():
    return 'Hello, World!'

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

