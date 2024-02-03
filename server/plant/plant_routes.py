# app/api/plant/routes/plant_routes.py
from flask import Blueprint
from plant.plant_controller import get_plants, get_plant_by_id

plant_bp = Blueprint("plant", __name__)

@plant_bp.route('/', methods=['GET'])
def plants():
    return get_plants()

@plant_bp.route('/<int:plant_id>', methods=['GET'])
def plant_by_id(plant_id):
    return get_plant_by_id(plant_id)

@plant_bp.route('/diseases', methods=['GET'])
def diseases():
    return get_diseases()


