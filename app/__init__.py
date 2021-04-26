from flask import Flask
from src.stylegan_lib import StyleGANServer

# define Flask app
app = Flask(__name__)

# define StyleGAN2 server
stgan_server = StyleGANServer()

# TODO: define pose server
pose_server = None

# import app routes
from app import routes
