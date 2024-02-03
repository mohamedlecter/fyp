import tensorflow as tf
import json
import os

class PlantDiseaseModel:
    def __init__(self, model_path='model/models/plant_disease_detector', class_names_path='model/class_names.json'):
        # Get the absolute paths of the model and class names files
        model_path = os.path.abspath(model_path)
        class_names_path = os.path.abspath(class_names_path)

        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)

        # Load class names from JSON file
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
            
    def predict(self, input_image):
        # Implement prediction logic using the built-in predict function
        prediction = self.model.predict(input_image)
        return prediction
    
    def get_class_name(self, class_index):
        return self.class_names[class_index]