import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def index():
    subs = sorted(os.listdir(app.config['ROOT']))
    app.config['subs'] = subs
    app.config['imgnames'] = {}
    filenames = sorted(os.listdir(os.path.join(app.config['ROOT'], subs[0])))
    app.config['imgnames'][subs[0]] = filenames
    return render_template('image_viewer.html', folder_name=subs[0], num_images=len(filenames))

@app.route('/send_first_image/<folder_name>')
def send_first_image(folder_name):
    filename = app.config['imgnames'][folder_name][0]
    return send_from_directory(os.path.join(app.config['ROOT'], folder_name), filename)

@app.route('/send_i_image/<string:folder_name>/<int:index>')
def send_i_image(folder_name, index):
    print(folder_name, index)
    return send_from_directory(os.path.join(app.config['ROOT'], folder_name), app.config['imgnames'][folder_name][index])

