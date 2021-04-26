from flask import jsonify, request
from app import app, stgan_server


@app.route('/tgenerate', methods=['POST'])
def tgenerate():
    if request.method == 'POST':
        pass


@app.route('/vgenerate', methods=['POST'])
def vgenerate():
    if request.method == 'POST':
        pass


@app.route('/refine', methods=['POST'])
def refine():
    if request.method == 'POST':
        pass


@app.route('/rotate', methods=['POST'])
def rotate():
    if request.method == 'POST':
        pass
