import requests

class PlantModel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://perenual.com/api/species-list"
        self.disease_url = "https://perenual.com/api/pest-disease-list" 
        

    def get_plants(self):
        # get Plant List
        response = requests.get(f"{self.base_url}?key={self.api_key}")
        plants = response.json()
        return plants

    def get_plant_by_id(self, plant_id):
        response = requests.get(f"{self.base_url}/{plant_id}?key={self.api_key}")
        plant_info = response.json()
        return plant_info
    
    def get_diseases(self):
        # get Disease List
        response = requests.get(f"{self.disease_url}?key={self.api_key}")
        diseases = response.json()
        return diseases
    
    def get_disease_by_id(self, disease_id):
        response = requests.get(f"{self.disease_url}/{disease_id}?key={self.api_key}")
        disease_info = response.json()
        return disease_info
