import os
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder='templates', static_folder='static')

IMAGE_FOLDER = os.path.join('static', 'data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:folder>')
def get_image_list(folder):
    # 获取指定文件夹内的所有图片文件名
    print('try to get', folder)
    subs = sorted(os.listdir(os.path.join(IMAGE_FOLDER, folder, 'images')))
    image_files = []
    for sub in subs:
        imgnames = sorted(os.listdir(os.path.join(IMAGE_FOLDER, folder, 'images', sub)))
        for imgname in imgnames:
            image_files.append(os.path.join(IMAGE_FOLDER, folder, 'images', sub, imgname))
        break
    print('get', len(image_files), 'images')
    return jsonify(image_files)

@app.route('/images/<path:folder>/<filename>')
def get_image(folder, filename):
    # 发送请求的图片文件
    return send_from_directory(os.path.join(IMAGE_FOLDER, folder), filename)

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    data = request.json
    print(data)  # 这里仅打印数据，实际应用中你可能需要将其存储在数据库中
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True, port=3456, host='0.0.0.0')
