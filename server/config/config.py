from pymongo import MongoClient
from flask_bcrypt import Bcrypt

class Config:
    MONGODB_CONNECTION_STRING = 'mongodb+srv://admin:admin@fyp.sabqf4f.mongodb.net/?retryWrites=true&w=majority'
    client = MongoClient(MONGODB_CONNECTION_STRING)
    db = client.get_database('fyp')
    bcrypt = Bcrypt()