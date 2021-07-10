from flask import Flask
from flask_cors import CORS
from src.stylegan_lib import StyleGANServer


# define Flask app
app = Flask(__name__)

# apply request cross origin
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# define StyleGAN2 server
stgan_server = StyleGANServer()

# TODO: define pose server
pose_server = None

# import app routes
from app import routes
