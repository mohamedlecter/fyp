from google.cloud import storage
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np

model = None
class_names = ["Apple__black_rot", "Apple__healthy", "Apple__rust", "Apple__scab", "Cassava__bacterial_blight", "Cassava__brown_streak_disease", "Cassava__green_mottle", "Cassava__healthy", "Cassava__mosaic_disease", "Corn__common_rust", "Corn__gray_leaf_spot", "Corn__healthy", "Grape__black_measles", "Grape__black_rot", "Grape__healthy", "Grape__leaf_blight_(isariopsis_leaf_spot)", "Peach__bacterial_spot", "Peach__healthy", "Pepper_bell__bacterial_spot", "Pepper_bell__healthy", "Pomegranate__diseased", "Pomegranate__healthy", "Potato__early_blight", "Potato__healthy", "Potato__late_blight", "Strawberry___leaf_scorch", "Strawberry__healthy", "Tomato__bacterial_spot", "Tomato__early_blight", "Tomato__healthy", "Tomato__late_blight", "Tomato__leaf_mold", "Tomato__mosaic_virus", "Tomato__septoria_leaf_spot", "Tomato__spider_mites_(two_spotted_spider_mite)", "Tomato__target_spot", "Tomato__yellow_leaf_curl_virus"]
BUCKET_NAME = "final-year-project-model"

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def load_model_from_bucket():
    global model
    if model is None:
        model_path = "/tmp/plant_disease_detector.h5"
        download_blob(BUCKET_NAME, "models/plant_disease_detector.h5", model_path)
        model = tf.keras.models.load_model(model_path)


def predict(request):
    global model
    if model is None:
        load_model_from_bucket()

    if 'file' not in request.files:
        return "No image file provided."

    image = request.files['file']

    if image.filename == '':
        return "No selected file."

    if image:
        
        # Load and preprocess the input image
        input_image = Image.open(image).convert("RGB").resize((224, 224))
        input_image = img_to_array(input_image)
        input_image = preprocess_input(input_image)
        input_image = np.expand_dims(input_image, axis=0)  # Add a batch dimension

        # Make predictions
        predictions = model.predict(input_image)

        # Get the predicted class
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        # Return the result
        result = {'class': predicted_class}

        return result
