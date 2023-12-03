import os
from os.path import join
from flask import Flask, render_template, jsonify
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def vis_keypoints3d():
    return render_template('index_smpl.html')

@app.route('/query_smpl')
def query_smpl():
    keypoints_data = []
    for filename in filenames:
        with open(join(root, filename), 'r') as f:
            keypoints_data.append(json.load(f))
    return jsonify(keypoints_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    root = args.root
    assert os.path.isdir(root), root
    filenames = sorted(os.listdir(root))
    print('Found {} files in {}'.format(len(filenames), root))

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)