from flask import Blueprint
from user.user_routes import user_bp

user_bp = Blueprint("user", __name__)

