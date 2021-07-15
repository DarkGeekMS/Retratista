from flask import jsonify, request
from flask_cors import cross_origin
import numpy as np

from app.rotator import app, pose_server
from app.utils import get_response_image


@app.route('/rotate', methods=['POST'])
@cross_origin()
def tgenerate():
    if request.method == 'POST':
        pass
