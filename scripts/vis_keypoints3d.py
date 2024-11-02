import os
from os.path import join
from flask import Flask, render_template, jsonify
import json
import numpy as np

template_folder = os.path.join(os.path.dirname(__file__), '..', 'easyannot', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'easyannot', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def vis_keypoints3d():
    return render_template('index_keypoints3d.html')


@app.route('/query_keypoints')
def query_keypoints():
    return jsonify(keypoints_data)

def read_keypoints_from_json(dirname):
    filenames = sorted(os.listdir(dirname))
    keypoints_data = []
    for filename in filenames:
        with open(join(dirname, filename), 'r') as f:
            keypoints_data.append(json.load(f))
    return keypoints_data

def rotate_keypoints(keypoints, rot):
    axis, angle = rot[0], np.deg2rad(rot[1])
    if axis == 'x':
        R = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    else:
        raise ValueError('Invalid rotation axis: {}'.format(axis))
    return keypoints @ R.T

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    root = args.root
    if os.path.isfile(root):
        if root.endswith('.pth'):
            rot = ('x', 90)
            edges = [
                [ 0, 1 ],
                [ 0, 2 ],
                [ 0, 3 ],
                [ 1, 4 ],
                [ 2, 5 ],
                [ 3, 6 ],
                [ 4, 7 ],
                [ 5, 8 ],
                [ 6, 9 ],
                [ 7, 10],
                [ 8, 11],
                [ 9, 12],
                [ 9, 13],
                [ 9, 14],
                [12, 15],
                [13, 16],
                [14, 17],
                [16, 18],
                [17, 19],
                [18, 20],
                [19, 21]
            ]
            import torch
            data = torch.load(root)
            keypoints = data['pred'].numpy()
            # rotate keypoints
            keypoints = rotate_keypoints(keypoints, rot)
            keypoints = keypoints.tolist()
            keypoints_data = [[{'id': 0, 'keypoints3d': keypoints[i], 'edges': edges}] for i in range(len(keypoints))]
    else:
        # keypoints: (seqlen, 22, 3)
        assert os.path.isdir(root), root
        keypoints_data = read_keypoints_from_json(root)

    print('Found {} files in {}'.format(len(keypoints_data), root))

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)