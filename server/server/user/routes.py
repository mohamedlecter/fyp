# # from flask import Flask, request, render_template, jsonify
# # from server import app
# # from user.models import User

# from flask import Blueprint, request, jsonify
# from user.models import User, bcrypt  # Importing User and bcrypt from the same package
# from server import mongo


# user_bp = Blueprint('user', __name__)

# @user_bp.route('/user/signup', methods=['POST'])
# def register():
#     username = request.json.get('username')
#     password = request.json.get('password')
#     email = request.json.get('email')
    

#     if not username or not password:
#         return jsonify({'error': 'Username and password are required'}), 400

#     existing_user = mongo.db.users.find_one({'username': username})
#     if existing_user:
#         return jsonify({'error': 'Username already exists'}), 400

#     new_user = User(username, password, email)
#     mongo.db.users.insert_one({'username': new_user.username, 'password': new_user.password, 'email': new_user.email})

#     return jsonify({'message': 'Registration successful'}), 201

# @user_bp.route('/user/login', methods=['POST'])
# def login():
#     email = request.json.get('email')
#     password = request.json.get('password')
    

#     if not email or not password:
#         return jsonify({'error': 'email and password are required'}), 400

#     user = mongo.db.users.find_one({'email': email})

#     if user and bcrypt.check_password_hash(user['password'], password):
#         # You can implement token-based authentication here
#         return jsonify({'message': 'Login successful'}), 200
#     else:
#         return jsonify({'error': 'Invalid credentials'}), 401
    
# @user_bp.route('/user/', methods=['GET'])
# def get_all_users():
#     users = mongo.db.users.find()
#     user_list = []
#     for user in users:
#         user_list.append({
#             'username': user['username'],
#             'email': user['email']
#         })
#     return jsonify(user_list), 200