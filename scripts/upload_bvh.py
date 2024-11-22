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
    return render_template('upload_keypoints3d.html')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)