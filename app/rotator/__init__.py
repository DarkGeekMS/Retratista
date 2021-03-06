from flask import Flask
from flask_cors import CORS
from src.pose_lib import PoseServer


# define Flask app
app = Flask(__name__)

# apply request cross origin
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# define poses server
pose_server = PoseServer()

# import app routes
from app.rotator import routes
