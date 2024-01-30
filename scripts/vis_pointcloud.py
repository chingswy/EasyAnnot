import os
from os.path import join
from flask import Flask, render_template, jsonify
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def vis_keypoints3d():
    return render_template('index_pointcloud.html')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    root = args.root

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)