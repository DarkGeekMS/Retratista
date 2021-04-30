from flask import jsonify, request
import numpy as np

from app import app, stgan_server, pose_server
from .utils import get_response_image


@app.route('/tgenerate', methods=['POST'])
def tgenerate():
    if request.method == 'POST':
        # get required face textual description
        content = request.get_json()
        sent = content.get('text')
        # perform text processing
        values = stgan_server.process_text(sent)
        # generate target face and get its attributes values
        face_image, values = stgan_server.generate_face(values)
        # encode output face image for response
        encoded_face = get_response_image(face_image)
        # return response JSON with output face and its values
        return jsonify({
            'face': encoded_face,
            'values': list(values)
        })


@app.route('/vgenerate', methods=['POST'])
def vgenerate():
    if request.method == 'POST':
        # get required facial attributes values
        content = request.get_json()
        values = np.array(content.get('values'))
        # generate target face and get its attributes values
        face_image, values = stgan_server.generate_face(values)
        # encode output face image for response
        encoded_face = get_response_image(face_image)
        # return response JSON with output face and its values
        return jsonify({
            'face': encoded_face,
            'values': list(values)
        })


@app.route('/refine', methods=['POST'])
def refine():
    if request.method == 'POST':
        # get required facial attribute and morph change
        content = request.get_json()
        type = content.get('type')
        idx = content.get('index')
        offset = content.get('offset')
        # refine generated face with given offset
        face_image = stgan_server.refine_face(type, idx, offset)
        # encode output face image for response
        encoded_face = get_response_image(face_image)
        # return response JSON with output face
        return jsonify({'face': encoded_face})


@app.route('/rotate', methods=['POST'])
def rotate():
    if request.method == 'POST':
        pass
