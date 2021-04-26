from flask import Flask
from src.stylegan_lib.stylegan2_server import StyleGANServer


app = Flask(__name__)

stgan_server = StyleGANServer()
