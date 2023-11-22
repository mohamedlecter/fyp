# app/api/user/routes/user_routes.py
from flask import Blueprint
from user.user_controllers import register_user, login_user, get_all_users, get_user_by_id, delete_user_by_id, update_user_by_id

user_bp = Blueprint("user", __name__)

@user_bp.route('/signup', methods=['POST'])
def register():
    return register_user()

@user_bp.route('/login', methods=['POST'])
def login():
    return login_user()

@user_bp.route('/', methods=['GET'])
def get_all():
    return get_all_users()

@user_bp.route('/<user_id>', methods=['GET'])
def get_by_id(user_id):
    return get_user_by_id(user_id)

@user_bp.route('/<user_id>', methods=['DELETE'])
def delete_by_id(user_id):
    return delete_user_by_id(user_id)

@user_bp.route('/<user_id>', methods=['PUT'])
def update_by_id(user_id):
    return update_user_by_id(user_id)

