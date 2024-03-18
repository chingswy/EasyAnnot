import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, redirect
from ..mytools.file_utils import read_yaml, write_yaml

template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def index():
    root = app.config['IMAGE_ROOT']
    print(f'Try to listdir of {root}')
    subs = sorted(os.listdir(root))
    print(subs)
    app.config['subs'] = subs
    app.config['imgnames'] = {}
    for sub in subs:
        print(f'- Try to listdir of {sub}')
        filenames = sorted(os.listdir(os.path.join(root, sub)))
        app.config['imgnames'][sub] = filenames
    filename = os.path.join(app.config['ROOT'], 'vanish.yml')
    app.config['results'] = read_yaml(filename)
    first_images = [(sub, app.config['imgnames'][sub][0]) for sub in subs]
    # return render_template('index_match_points.html', first_images=first_images)
    return render_template('index_any.html', first_images=first_images, href='annot_vanish')


@app.route('/images/<string:folder_name>/<string:filename>')
def send_image(folder_name, filename):
    root = app.config['IMAGE_ROOT']
    print(root, filename)
    return send_from_directory(os.path.join(root, folder_name), os.path.basename(filename))


@app.route('/send_i_image/<string:folder_name>/<int:index>')
def send_i_image(folder_name, index):
    return send_from_directory(os.path.join(app.config['IMAGE_ROOT'], folder_name), app.config['imgnames'][folder_name][index])

@app.route('/send_first_image/<folder_name>')
def send_first_image(folder_name):
    return send_i_image(folder_name, 0)

@app.route('/annot_vanish/<string:folder_name>')
def annot_vanish(folder_name):
    if 'imgnames' not in app.config:
        return redirect('/')
    return render_template('annot_vanish.html', folder_name=folder_name, num_images=len(app.config['imgnames'][folder_name]))

@app.route('/submit_vanish/<string:folder_name>', methods=['POST'])
def submit_vanish(folder_name):
    lines = request.json['lines']
    record = app.config['results']
    record[folder_name] = lines
    filename = os.path.join(app.config['ROOT'], 'vanish.yml')
    # calculate the intersect points
    for direct in ['X', 'Y', 'Z']:
        line = record[folder_name][direct]
        if len(line) < 3: continue
        points = np.array([[[l['startX'], l['startY']], [l['endX'], l['endY']]] for l in line]).transpose(1, 0, 2)
        vanish = calc_vanishpoint(points)
        record[folder_name][direct + '_point'] = vanish[:2].tolist()
    print(record.keys())
    write_yaml(filename, app.config['results'])
    return jsonify({"status": "success"})

@app.route('/query_vanish/<string:folder_name>', methods=['GET'])
def query_vanish(folder_name):
    record = app.config['results']
    if folder_name not in record or len(record[folder_name].keys()) == 0:
        record[folder_name] = {'X': [], 'Y': [], 'Z': []}
    # 处理 lines 数据
    return jsonify(record[folder_name])

def calc_vanishpoint(keypoints2d):
    '''
        keypoints2d: (2, N, 2)
    '''
    # weight: (N, 1)
    A = np.hstack([
        keypoints2d[1, :, 1:2] - keypoints2d[0, :, 1:2],
        -(keypoints2d[1, :, 0:1] - keypoints2d[0, :, 0:1])
    ])
    b = -keypoints2d[0, :, 0:1]*(keypoints2d[1, :, 1:2] - keypoints2d[0, :, 1:2]) \
        + keypoints2d[0, :, 1:2] * (keypoints2d[1, :, 0:1] - keypoints2d[0, :, 0:1])
    b = -b
    avgInsec = np.linalg.inv(A.T @ A) @ (A.T @ b)
    result = np.zeros(3)
    result[0] = avgInsec[0, 0]
    result[1] = avgInsec[1, 0]
    return result

def focal_from_points(point0, point1, height, width):
    vanish_point = np.stack([np.array(point0), np.array(point1)])
    K = np.eye(3)
    H = height
    W = width
    vanish_point[:, 0] -= W/2
    vanish_point[:, 1] -= H/2
    focal = np.sqrt(-(vanish_point[0][0]*vanish_point[1][0] + vanish_point[0][1]*vanish_point[1][1]))
    return focal
    
@app.route('/calculateFocal/<string:folder_name>', methods=['GET'])
def calculateFocal(folder_name):
    record = app.config['results'][folder_name]
    if 'X_point' not in record or 'Y_point' not in record:
        return jsonify({'focal': 0})
    xpoint = record['X_point']
    ypoint = record['Y_point']
    imgname0 = app.config['imgnames'][folder_name][0]
    imgname0 = os.path.join(app.config['IMAGE_ROOT'], folder_name, imgname0)
    img = cv2.imread(imgname0)
    height, width = img.shape[:2]
    focal = focal_from_points(xpoint, ypoint, height, width)
    print(focal)
    record['focal'] = float(focal)
    return jsonify({'focal': focal})