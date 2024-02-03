from flask import jsonify
from plant.plant_model import PlantModel

plant_model = PlantModel(api_key='sk-PFs2655e223d2d8613080')

def get_plants():
    plants = plant_model.get_plants()
    return jsonify(plants)

def get_plant_by_id(plant_id):
    plant_info = plant_model.get_plant_by_id(plant_id)
    return jsonify(plant_info)

def get_diseases():
    diseases = plant_model.get_diseases()
    return jsonify(diseases)


def get_disease_by_id(disease_id):
    disease_info = plant_model.get_disease_by_id(disease_id)
    return jsonify(disease_info)

