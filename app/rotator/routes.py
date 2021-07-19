from flask import jsonify, request
from flask_cors import cross_origin
import io
import base64
import numpy as np
from PIL import Image

from app.rotator import app, pose_server
from app.utils import get_response_image


@app.route('/rotate', methods=['POST'])
@cross_origin()
def tgenerate():
    if request.method == 'POST':
        # get required image and angle of rotation
        content = request.get_json()
        image = content.get('image')
        angle = content.get('angle')
        # post-process required image
        image = base64.b64decode(str(image))
        image = Image.open(io.BytesIO(image))
        #image = image.resize((1024,1024), Image.ANTIALIAS)
        image = np.array(image) 
        image = image[:, :, ::-1].copy() 
        # rotate generated face with given angle
        face_image = pose_server.rotate_face(image, angle)
        # encode output face image for response
        encoded_face = get_response_image(face_image)
        # return response JSON with output face
        return jsonify({'face': encoded_face})
