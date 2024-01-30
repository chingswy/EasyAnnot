import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort


template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/images/<path:path>')
def send_images(path):
    root = app.config['ROOT']
    imgname = os.path.join(root, 'images', path)
    print(imgname)
    return send_from_directory(os.path.dirname(imgname), os.path.basename(imgname))

camera_location = np.array([
    [0., 0., 0.],
    [-1, -1, 0.5],
    [-1, 1., 0.5],
    [1., 1., 0.5],
    [1., -1., 0.5]
])

# query the camera positions
@app.route('/query_cameras', methods=['GET'])
def get_cameras():
    root = app.config['ROOT']
    from easymocap.mytools.camera_utils import read_cameras
    cameras = read_cameras(root)
    cameras_list = []
    for cam, camera in cameras.items():
        imgname = os.path.join('images', cam, '000000.jpg')
        R = camera['R']
        T = camera['T']
        center = -R.T @ T
        cam_loc = camera_location.copy() * 0.2
        location = (cam_loc - T.T) @ R.T.T
        print(location)
        rotation = R.T
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rotation
        center = center[:, 0].tolist()
        cameras_list.append({
            'name': cam,
            'center': center,
            'location': location.tolist(),
            'rotation': rot_4x4.tolist(),
            'imgname': imgname
        })
    return jsonify(cameras_list)