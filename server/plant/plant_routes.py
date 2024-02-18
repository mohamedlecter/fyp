# app/api/plant/routes/plant_routes.py
import re
from flask import Blueprint, jsonify
from plant.plant_controller import get_plants, get_plant_by_id, upload_file
import os


file = os.path.abspath("server\data.json")

plant_bp = Blueprint("plant", __name__)

@plant_bp.route('/', methods=['GET'])
def plants():
    return get_plants()

@plant_bp.route('/<plant_id>', methods=['GET'])
def plant_by_id(plant_id):
    return get_plant_by_id(plant_id)


@plant_bp.route('/upload_csv', methods=['POST'])
def upload_csv():
    return upload_file(file)

