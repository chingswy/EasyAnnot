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

def get_first_images(root_path):
    first_images = []
    for folder_name in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder_name)
        if os.path.isdir(folder_path):
            # Sort the images to ensure correct order
            images = sorted([img for img in os.listdir(
                folder_path) if img.endswith('.jpg')])
            if images:
                # Get the first image
                first_images.append((folder_name, images[0]))
    return first_images

@app.route('/gallery')
def gallery():
    first_images = get_first_images(os.path.join(ROOT, 'images'))
    return render_template('index_gallery.html', first_images=first_images)

# Route to display all images within a folder

@app.route('/folder/<folder_name>')
def show_folder(folder_name):
    folder_path = os.path.join(ROOT, 'images', folder_name)
    if os.path.isdir(folder_path):
        images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])
        return render_template('folder.html', folder_name=folder_name, images=images)
    else:
        return 'Folder not found', 404

# Route to serve images

@app.route('/images/<folder_name>/<filename>')
def send_image(folder_name, filename):
    print(folder_name, filename)
    return send_from_directory(os.path.join(ROOT, 'images', folder_name), filename)

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

# @app.route('/images/<path:folder>')
# def get_image_list(folder):
#     # 获取指定文件夹内的所有图片文件名
#     print('try to get', folder)
#     subs = sorted(os.listdir(os.path.join(IMAGE_FOLDER, folder, 'images')))
#     image_files = []
#     for sub in subs:
#         imgnames = sorted(os.listdir(os.path.join(
#             IMAGE_FOLDER, folder, 'images', sub)))
#         for imgname in imgnames:
#             image_files.append(os.path.join(
#                 IMAGE_FOLDER, folder, 'images', sub, imgname))
#         break
#     print('get', len(image_files), 'images')
#     return jsonify(image_files)


# @app.route('/images/<path:folder>/<filename>')
# def get_image(folder, filename):
#     # 发送请求的图片文件
#     return send_from_directory(os.path.join(IMAGE_FOLDER, folder), filename)


@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    print(data)  # 这里仅打印数据，实际应用中你可能需要将其存储在数据库中
    return jsonify(success=True)


if __name__ == '__main__':
    app.run(debug=True, port=3456, host='0.0.0.0')
