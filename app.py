import os
import shutil
from tqdm import tqdm
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
import yaml

app = Flask(__name__, template_folder='templates', static_folder='static')

IMAGE_FOLDER = os.path.join('static', 'data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index3d')
def index3d():
    return render_template('index_bvh.html')

# Index route to display the first image of each folder

# Function to get the first image from each folder

def read_yaml(filename):
    if not os.path.exists(filename):
        if input(f'File {filename} not found, do you want to create a new one? (y/n)') == 'y':
            write_yaml(filename, {})
        else:
            return 0
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def write_yaml(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        yaml.dump(data, f)

def prepare_dataset(root):
    filenames = {}
    subs = sorted(os.listdir(os.path.join(root, IMAGES)))
    for sub in tqdm(subs):
        filenames[sub] = sorted(os.listdir(os.path.join(root, IMAGES, sub)))
    return filenames

def get_first_images():
    first_images = [(sub, filename[0]) for sub, filename in filenames.items()]
    return first_images

if True:
    # 获取图像数据库的基本操作
    # Route to display all images within a folder
    @app.route('/folder/<folder_name>')
    def show_folder(folder_name):
        folder_path = os.path.join(ROOT, IMAGES, folder_name)
        if os.path.isdir(folder_path):
            images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])
            return render_template('image_folder.html', folder_name=folder_name, images=images)
        else:
            return 'Folder not found', 404

    @app.route('/images/<string:folder_name>/<string:filename>')
    def send_image(folder_name, filename):
        return send_from_directory(os.path.join(ROOT, IMAGES, folder_name), filename)

    @app.route('/send_i_image/<string:folder_name>/<int:index>')
    def send_i_image(folder_name, index):
        return send_from_directory(os.path.join(ROOT, IMAGES, folder_name), filenames[folder_name][index])

    @app.route('/send_first_image/<folder_name>')
    def send_first_image(folder_name):
        filename = filenames[folder_name][0]
        return send_from_directory(os.path.join(ROOT, IMAGES, folder_name), filename)

    @app.route('/get_num_images/<folder_name>')
    def get_num_images(folder_name):
        return len(filenames[folder_name])

if True:
    # ROI的基本操作
    @app.route('/roi')
    def roi():
        first_images = get_first_images()
        return render_template('index_roi.html', first_images=first_images)
    
    @app.route('/annot_roi/<string:folder_name>')
    def annot_roi(folder_name):
        return render_template('annot_roi.html', folder_name=folder_name, num_images=len(filenames[folder_name]))
    
    @app.route('/query_roi/<string:folder_name>', methods=['GET'])
    def query_roi(folder_name):
        if os.path.exists(ROI_NAME):
            with open(ROI_NAME, 'r') as f:
                rois = json.load(f)
        else:
            rois = {}
            for sub in filenames.keys():
                rois[sub] = []
            with open(ROI_NAME, 'w') as f:
                json.dump(rois, f, indent=4)
        if folder_name not in rois:
            rois[folder_name] = []
            with open(ROI_NAME, 'w') as f:
                json.dump(rois, f, indent=4)
        return jsonify(rois[folder_name])
    
    @app.route('/submit_roi/<string:folder_name>', methods=['POST'])
    def submit_roi(folder_name):
        L = int(request.form['L'])
        T = int(request.form['T'])
        R = int(request.form['R'])
        B = int(request.form['B'])
        with open(ROI_NAME, 'r') as f:
            rois = json.load(f)
        rois[folder_name] = [L, T, R, B]
        with open(ROI_NAME, 'w') as f:
            json.dump(rois, f, indent=4)
        return render_template('annot_roi.html', folder_name=folder_name, num_images=len(filenames[folder_name]))

if True:
    # 标注匹配点的操作
    @app.route('/match_points')
    def match_points():
        first_images = get_first_images()
        return render_template('index_match_points.html', first_images=first_images)
    
    @app.route('/query_points', methods=['GET'])
    def query_points():
        if os.path.exists(POINTS_NAME):
            with open(POINTS_NAME, 'r') as f:
                points = json.load(f)
        else:
            points = {}
            for sub in filenames.keys():
                points[sub] = []
            with open(POINTS_NAME, 'w') as f:
                json.dump(points, f)
        return jsonify(points)

    @app.route('/export_points', methods=['POST'])
    def export_points():
        data = request.get_json()
        # 处理数据...
        with open(POINTS_NAME, 'w') as f:
            json.dump(data, f, indent=4)
        return jsonify({"status": "success", "message": "Marks data received"})

    @app.route('/triangulate', methods=['POST'])
    def triangulate():
        datas = request.get_json()
        print(datas)
        import numpy as np
        from easymocap.mytools.camera_utils import read_cameras
        cameras = read_cameras(os.path.join(ROOT, CAMERAS))
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
        from library.triangulate import batch_triangulate, project_wo_dist, project_w_dist
        k3d = batch_triangulate(records2d, Pall)
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

@app.route('/gallery')
def gallery():
    first_images = get_first_images()
    return render_template('index_gallery.html', first_images=first_images)

@app.route('/synchronize')
def synchronize():
    first_images = get_first_images(os.path.join(ROOT, IMAGES))
    return render_template('index_gallery.html', first_images=first_images)

if True:
    @app.route('/clip')
    def clip():
        subs = list(filenames.keys())
        num_images = len(filenames[subs[0]])
        if os.path.exists(CLIP_NAME):
            with open(CLIP_NAME, 'r') as f:
                clips = json.load(f)
        else:
            clips = []
            with open(CLIP_NAME, 'w') as f:
                json.dump(clips, f)
        return render_template('annot_clip.html', subs=subs, num_images=num_images, clips=clips)

    @app.route('/save_clips', methods=['POST'])
    def save_clips():
        data = request.json
        clips = data.get('clips')
        with open(CLIP_NAME, 'w') as f:
            json.dump(clips, f, indent=4)
        # 这里添加处理clips的逻辑，比如保存到数据库或文件
        # 例如：save_to_database(clips)
        return jsonify({'status': 'success', 'message': 'Clips data received and processed'}), 200
    
    @app.route('/copy_clips', methods=['GET'])
    def copy_clips():
        with open(CLIP_NAME, 'r') as f:
            clips = json.load(f)
        for clip in clips:
            start, end = clip['start'], clip['end']
            outdir = os.path.join(ROOT, 'clips', f'{start}_{end}', IMAGES)
            subs = sorted(os.listdir(os.path.join(ROOT, IMAGES)))
            print(clip)
            for sub in subs:
                outdir_sub = os.path.join(outdir, sub)
                os.makedirs(outdir_sub, exist_ok=True)
                for new_frame, frame in enumerate(tqdm(range(start, end+1))):
                    filename = filenames[sub][frame]
                    dstname = os.path.join(outdir_sub, f'{new_frame:06d}.jpg')
                    srcname = os.path.join(ROOT, IMAGES, sub, filename)
                    if os.path.exists(dstname):
                        continue
                    shutil.copy(srcname, dstname)

@app.route('/annotations/<folder_name>/<image_name>')
def get_annotations(folder_name, image_name):
    # Construct the file path to the JSON file.
    json_file_path = os.path.join(ROOT, 'annots', folder_name, f'{image_name.replace(".jpg", "")}.json')
    # Check if the file exists.
    if os.path.exists(json_file_path) and os.path.isfile(json_file_path):
        # Open the JSON file and return its contents.
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            # If there's an error reading the file, return a server error.
            print(f"Error reading file: {e}")
            abort(500)
    else:
        # If the file does not exist, return a 404 not found error.
        abort(404)

@app.route('/keypoints3d/<folder_name>/<image_name>')
def get_keypoints3d(folder_name, image_name):
    # Construct the file path to the JSON file.
    json_file_path = os.path.join(ROOT, 'output', 'keypoints3d', folder_name, f'{image_name.replace(".jpg", "")}.json')
    print(json_file_path)
    # Check if the file exists.
    if os.path.exists(json_file_path) and os.path.isfile(json_file_path):
        # Open the JSON file and return its contents.
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            # If there's an error reading the file, return a server error.
            print(f"Error reading file: {e}")
            abort(500)
    else:
        # If the file does not exist, return a 404 not found error.
        abort(404)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    print(data)  # 这里仅打印数据，实际应用中你可能需要将其存储在数据库中
    return jsonify(success=True)

if True: # vanish points
    @app.route('/vanish')
    def vanish():
        first_images = get_first_images()
        return render_template('index_any.html', first_images=first_images, href='annot_vanish')

    @app.route('/annot_vanish/<string:folder_name>')
    def annot_vanish(folder_name):
        return render_template('annot_vanish.html', folder_name=folder_name, num_images=len(filenames[folder_name]))
    
    @app.route('/submit_vanish/<string:folder_name>', methods=['POST'])
    def submit_vanish(folder_name):
        lines = request.json['lines']
        record = read_yaml(VANISH_NAME)
        record[folder_name] = lines
        write_yaml(VANISH_NAME, record)
        # 处理 lines 数据
        return jsonify({"status": "success"})

    @app.route('/query_vanish/<string:folder_name>', methods=['GET'])
    def query_vanish(folder_name):
        record =read_yaml(VANISH_NAME)
        if folder_name not in record or len(record[folder_name].keys()) == 0:
            record[folder_name] = {'X': [], 'Y': [], 'Z': []}
        # 处理 lines 数据
        return jsonify(record[folder_name])    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument('--port', type=int, default=3456)
    parser.add_argument('--readonly', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ROOT = args.root
    IMAGES = args.images
    CAMERAS = ''

    if not args.readonly:
        try:
            if os.path.exists(os.path.join(ROOT, '_annotations')):
                shutil.rmtree(os.path.join(ROOT, '_annotations'))
            os.makedirs(os.path.join(ROOT, '_annotations'))
        except PermissionError:
            print('Permission denied, please check the path')
            exit()
    # check if this file is updated
    # if not, seems error, exit
    # if yes, continue
    filenames = prepare_dataset(ROOT)
    CLIP_NAME = os.path.join(ROOT, 'clips.json')
    MV_CLIP_NAME = os.path.join(ROOT, 'clips_mv.json')
    ROI_NAME = os.path.join(ROOT, 'roi.json')
    POINTS_NAME = os.path.join(ROOT, 'match_points.json')
    VANISH_NAME = os.path.join(ROOT, 'vanish_points.yml')

    app.run(debug=args.debug, port=args.port, host='0.0.0.0')