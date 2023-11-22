from flask import Flask
from flask_cors import CORS
from config.config import Config
from user.user_routes import user_bp
from diagnose.diagnose_routes import diagnose_bp

app = Flask(__name__)
CORS(app)


# Load configuration
app.config.from_object(Config)

# Initialize MongoDB client and Bcrypt
client = Config.client
db = Config.db
bcrypt = Config.bcrypt


# Register Blueprints
app.register_blueprint(user_bp, url_prefix='/user')
app.register_blueprint(diagnose_bp, url_prefix='/diagnose')

# # Load the trained model
# model = tf.keras.models.load_model('../model 2/models/plant_disease_detector')

# @app.route('/diagnose', methods=['POST'])
# def diagnose():
#     if 'image' not in request.files:
#         return "No image file provided."

#     image = request.files['image']

#     if image.filename == '':
#         return "No selected file."

#     if image:
#         # Save the uploaded image to a temporary file
#         image_path = 'temp.jpg'
#         image.save(image_path)

#         # Load and preprocess the input image
#         input_image = load_img(image_path, target_size=(224, 224))
#         input_image = img_to_array(input_image)
#         input_image = preprocess_input(input_image)
#         input_image = np.expand_dims(input_image, axis=0)  # Add a batch dimension

#         # Make predictions
#         predictions = model.predict(input_image)

#         # Get the predicted class
#         predicted_class_index = np.argmax(predictions)
#         predicted_class = class_names[predicted_class_index]

#         # Clean up temporary file
#         os.remove(image_path)

#         return f'Predicted class: {predicted_class}'



if __name__ == '__main__':
    app.run(debug=True)

