import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def index():
    image_root = app.config['IMAGE_ROOT']
    subs = sorted(os.listdir(image_root))
    app.config['subs'] = subs
    app.config['imgnames'] = {}
    filenames = sorted(os.listdir(os.path.join(image_root, subs[0])))
    app.config['imgnames'][subs[0]] = filenames
    return render_template('annot_viewer.html', folder_name=subs[0], num_images=len(filenames))

@app.route('/send_first_image/<folder_name>')
def send_first_image(folder_name):
    image_root = app.config['IMAGE_ROOT']
    filename = app.config['imgnames'][folder_name][0]
    print(image_root, folder_name, filename)
    return send_from_directory(os.path.join(image_root, folder_name), filename)

@app.route('/send_i_image/<string:folder_name>/<int:index>')
def send_i_image(folder_name, index):
    image_root = app.config['IMAGE_ROOT']
    return send_from_directory(os.path.join(image_root, folder_name), app.config['imgnames'][folder_name][index])

@app.route('/query_annots/<string:folder_name>/<int:index>')
def query_annots(folder_name, index):
    filename = app.config['imgnames'][folder_name][index]
    annot_root = app.config['ANNOT_ROOT']
    fullname = os.path.join(annot_root, folder_name, filename) + '.json'
    with open(fullname) as f:
        data = json.load(f)
    return jsonify(data)