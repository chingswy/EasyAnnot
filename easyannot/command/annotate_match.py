import os
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from ..library.triangulate import batch_triangulate, project_wo_dist, project_w_dist
import json

template_folder = os.path.join(os.path.dirname(__file__), '..', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def index():
    root = app.config['IMAGE_ROOT']
    subs = sorted(os.listdir(root))
    app.config['subs'] = subs
    app.config['imgnames'] = {}
    for sub in subs:
        filenames = sorted(os.listdir(os.path.join(root, sub)))
        app.config['imgnames'][sub] = [filenames[app.config['frame']]]
    first_images = [(sub, os.path.join(sub, app.config['imgnames'][sub][0])) for sub in subs]
    return render_template('index_match_points.html', first_images=first_images)

@app.route('/send_i_image/<string:folder_name>/<int:index>')
def send_i_image(folder_name, index):
    return send_from_directory(os.path.join(app.config['IMAGE_ROOT'], folder_name), app.config['imgnames'][folder_name][index])

@app.route('/query_points', methods=['GET'])
def query_points():
    points_name = app.config['POINTS_NAME']
    if os.path.exists(points_name):
        with open(points_name, 'r') as f:
            points = json.load(f)
    else:
        points = {sub: [] for sub in app.config['subs']}
    return jsonify(points)

@app.route('/export_points', methods=['POST'])
def export_points():
    data = request.get_json()
    # 处理数据...
    points_name = app.config['POINTS_NAME']
    with open(points_name, 'w') as f:
        json.dump(data, f, indent=4)
    return jsonify({"status": "success", "message": "Marks data received"})

@app.route('/triangulate', methods=['POST'])
def triangulate():
    datas = request.get_json()
    print(datas)
    import numpy as np
    from easymocap.mytools.camera_utils import read_cameras
    cameras = read_cameras(os.path.join(app.config['ROOT'], app.config['CAMERA']))
    # 阅读匹配点
    # 阅读相机参数
    # 三角化
    cams = list(cameras.keys())
    Pall = np.stack([cameras[c]['P'] for c in cams], axis=0)
    records2d = np.zeros((len(cams), 1000, 3))
    colors = {}
    max_id = -1
    for nv, sub in enumerate(cams):
        data = datas[sub]
        for annot in data:
            pid = annot['id']
            if pid > max_id:
                max_id = pid
            if pid not in colors:
                colors[pid] = annot['color']
            x, y = annot['x'], annot['y']
            records2d[nv, pid, 0] = x
            records2d[nv, pid, 1] = y
            records2d[nv, pid, 2] = 1.
    # (nViews, nPoints, 3)
    records2d = records2d[:, :max_id+1, :]
    from easymocap.mytools.camera_utils import Undistort
    for nv in range(records2d.shape[0]):
        records2d[nv] = Undistort.points(records2d[nv], cameras[cams[nv]]['K'], cameras[cams[nv]]['dist'])
    k3d = batch_triangulate(records2d, Pall)
    for i in range(k3d.shape[0]):
        for j in range(k3d.shape[1]):
            print(k3d[i, j], end=', ')
        print('')
    # k2d_repro: (nViews, nPoints, 3: (x, y, conf))
    k2d_repro, depth = project_wo_dist(k3d, Pall)
    for nv in range(records2d.shape[0]):
        camera = {key: cameras[cams[nv]][key] for key in ['K', 'R', 'T', 'dist']}
        _k2d_repro, _depth = project_w_dist(k3d, camera)
        k2d_repro[nv] = _k2d_repro
    ret = {}
    for i, sub in enumerate(cams):
        ret[sub] = []
        for j in range(k3d.shape[0]):
            ret[sub].append({
                'id': j,
                'x': float(k2d_repro[i, j, 0]),
                'y': float(k2d_repro[i, j, 1]),
                'color': colors[j]
            })
    return jsonify(ret)
