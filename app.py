import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

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
ROOT = '/mnt/v01/clips/HuMiD-yukagawa-clips'
ROOT = os.environ['ROOT']

def prepare_dataset(root):
    filenames = {}
    subs = sorted(os.listdir(os.path.join(root, 'images')))
    for sub in subs:
        filenames[sub] = sorted(os.listdir(os.path.join(root, 'images', sub)))
    return filenames

filenames = prepare_dataset(ROOT)

def get_first_images():
    first_images = [(sub, filename[0]) for sub, filename in filenames.items()]
    return first_images

if True:
    # 获取图像数据库的基本操作
    # Route to display all images within a folder
    @app.route('/folder/<folder_name>')
    def show_folder(folder_name):
        folder_path = os.path.join(ROOT, 'images', folder_name)
        if os.path.isdir(folder_path):
            images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])
            return render_template('folder.html', folder_name=folder_name, images=images)
        else:
            return 'Folder not found', 404

    @app.route('/images/<string:folder_name>/<string:filename>')
    def send_image(folder_name, filename):
        return send_from_directory(os.path.join(ROOT, 'images', folder_name), filename)

    @app.route('/send_i_image/<string:folder_name>/<int:index>')
    def send_i_image(folder_name, index):
        return send_from_directory(os.path.join(ROOT, 'images', folder_name), filenames[folder_name][index])

    @app.route('/send_first_image/<folder_name>')
    def send_first_image(folder_name):
        filename = filenames[folder_name][0]
        return send_from_directory(os.path.join(ROOT, 'images', folder_name), filename)

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
    
    @app.route('/submit_roi', methods=['POST'])
    def submit_roi():
        x = request.form['x']
        y = request.form['y']
        width = request.form['width']
        height = request.form['height']

        # 在这里处理坐标数据
        # 例如，保存到数据库或执行其他操作

        return '坐标已提交：X={}, Y={}, Width={}, Height={}'.format(x, y, width, height)

if True:
    # 标注匹配点的操作
    @app.route('/match_points')
    def match_points():
        first_images = get_first_images()
        return render_template('index_match_points.html', first_images=first_images)
    
    POINTS_NAME = os.path.join(ROOT, 'match_points.json')

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

@app.route('/gallery')
def gallery():
    first_images = get_first_images()
    return render_template('index_gallery.html', first_images=first_images)

@app.route('/synchronize')
def synchronize():
    first_images = get_first_images(os.path.join(ROOT, 'images'))
    return render_template('index_gallery.html', first_images=first_images)

@app.route('/clip')
def clip():
    subs = list(filenames.keys())
    num_images = len(filenames[subs[0]])
    return render_template('clips.html', subs=subs, num_images=num_images)

@app.route('/save_clips', methods=['POST'])
def save_clips():
    data = request.json
    clips = data.get('clips')
    print(clips)
    # 这里添加处理clips的逻辑，比如保存到数据库或文件
    # 例如：save_to_database(clips)
    return jsonify({'status': 'success', 'message': 'Clips data received and processed'}), 200

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


if __name__ == '__main__':
    app.run(debug=True, port=3456, host='0.0.0.0')
