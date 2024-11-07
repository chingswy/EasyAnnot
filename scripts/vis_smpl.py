import os
from os.path import join
from flask import Flask, render_template, jsonify
import json

template_folder = os.path.join(os.path.dirname(__file__), '..', 'easyannot', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'easyannot', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

@app.route('/')
def vis_keypoints3d():
    return render_template('index_smpl.html')

@app.route('/query_smpl')
def query_smpl():
    return jsonify(smpl_data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    root = args.root
    smpl_data = []
    if os.path.isfile(root):
        if root.endswith('.pth'):
            import torch
            data = torch.load(root)
            data = {
                key: val.tolist() for key, val in data.items()
            }
            smpl_data = [[
                {
                    'id': 0,
                    'Rh': data['Rh'][i:i+1],
                    'Th': data['Th'][i:i+1],
                    'poses': data['poses'][i:i+1],
                    'shapes': data['shapes'][i:i+1]
                }
            ] for i in range(len(data['Rh']))
            ]
            print(f'Found {len(smpl_data)} frames in {root}')

        elif root.endswith('.pkl'):
            import ipdb; ipdb.set_trace()
    elif os.path.isdir(root):
        filenames = sorted(os.listdir(root))
        print('Found {} files in {}'.format(len(filenames), root))
    else:
        raise ValueError(f'Invalid root: {root}')

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)