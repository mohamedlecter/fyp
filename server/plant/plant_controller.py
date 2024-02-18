import json
import os
from bson import ObjectId
from flask import Flask, request, jsonify
from plant.plant_model import PlantModel
import csv
from config.config import Config
import pandas as pd

plant_model = PlantModel(api_key='sk-PFs2655e223d2d8613080')

bcrypt = Config.bcrypt
client = Config.client
db = Config.db

def get_plants():
    plants =  db.db.plants.find()
    plants_list = []
    for plant in plants:
        plants_list.append({
            "id": str(plant['_id']),
            "topic_questions": plant.get("topic_questions", ""),
            "summary": plant.get("summary", ""),
            "image": plant.get("image", ""),
            "link_underline": plant.get("link-underline", ""),
            "link": plant.get("link", ""),
            "title": plant.get("title", ""),
            "description": plant.get("description", ""),
            "scientific_name": plant.get("scientific_name", ""),
            "common_name": plant.get("common_name:", ""), 
            "uses": plant.get("uses", ""),
            "basic_requirements": plant.get("basic_requirements", ""),
            "growing": plant.get("growing", ""),
            "care": plant.get("care", ""),
            "harvesting": plant.get("harvesting", ""),
            "diseases": plant.get("diseases", [])
        })
    return jsonify(plants_list)


def get_plant_by_id(plant_id):
    plant = db.db.plants.find_one({"_id": ObjectId(plant_id)})
    if plant:
        return jsonify({
            "id": str(plant['_id']),
            "topic_questions": plant.get("topic_questions", ""),
            "summary": plant.get("summary", ""),
            "image": plant.get("image", ""),
            "link_underline": plant.get("link-underline", ""),
            "link": plant.get("link", ""),
            "title": plant.get("title", ""),
            "description": plant.get("description", ""),
            "scientific_name": plant.get("scientific_name", ""),
            "common_name": plant.get("common_name:", ""), 
            "uses": plant.get("uses", ""),
            "basic_requirements": plant.get("basic_requirements", ""),
            "growing": plant.get("growing", ""),
            "care": plant.get("care", ""),
            "harvesting": plant.get("harvesting", ""),
            "diseases": plant.get("diseases", [])
        })
    else:
        return jsonify({"error": "Plant not found"})

def upload_file(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': f"File '{file_path}' not found."})

        # Open the JSON file
        with open(file_path, 'r', encoding="utf8") as json_file:
            # Load JSON data from the file
            json_data = json.load(json_file)

            # Insert data into MongoDB collection
            db.db.plants.insert_many(json_data)

            return jsonify({'success': True, 'message': 'JSON data inserted to MongoDB'})
    except json.JSONDecodeError:
        return jsonify({'error': 'JSON decode error. The file might be empty or not in valid JSON format.'})
    except Exception as e:
        return jsonify({'error': str(e)})
