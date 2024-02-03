from flask import Flask
from flask_cors import CORS
from config.config import Config
from user.user_routes import user_bp
from diagnose.diagnose_routes import diagnose_bp
from plant.plant_routes import plant_bp

app = Flask(__name__)
CORS(app)


# Load configuration
app.config.from_object(Config)

# Initialize MongoDB client and Bcrypt
client = Config.client
db = Config.db
bcrypt = Config.bcrypt


# Register Blueprints
app.register_blueprint(user_bp, url_prefix='/user')
app.register_blueprint(diagnose_bp, url_prefix='/diagnose')
app.register_blueprint(plant_bp, url_prefix='/plant')



if __name__ == '__main__':
    app.run(debug=True)

