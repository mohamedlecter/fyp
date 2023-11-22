from flask import Blueprint
from diagnose.diagnose_controller import diagnose_plant

diagnose_bp = Blueprint("diagnose", __name__)

@diagnose_bp.route('/', methods=['POST'])
def diagnose():
    return diagnose_plant()
