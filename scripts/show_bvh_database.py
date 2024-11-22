import os
import glob
from os.path import join
from flask import Flask, render_template, jsonify
from flask import redirect, url_for
import json
import numpy as np
import urllib.parse

template_folder = os.path.join(os.path.dirname(__file__), '..', 'easyannot', 'templates')
static_folder = os.path.join(os.path.dirname(__file__), '..', 'easyannot', 'static')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

# ├── audio
# │   ├── zhuge_c0013_i1_001.wav
# ├── MoCap_bvh
# │   ├── zhuge_c0013_i1_001_rx.bvh
# ├── MoCap_bvh
files_info = []

def read_json_label(label_path):
    with open(label_path, 'r') as f:
        return json.load(f)

@app.route('/')
def list_mocap_files():
    root_path = args.root
    mocap_folder = os.path.join(root_path, 'MoCap_bvh')
    label_folder = os.path.join(root_path, 'labeling')
    audio_folder = os.path.join(root_path, 'audio')

    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
    for audio_file in audio_files:
        # try to find the bvh file
        bvh_file = glob.glob(os.path.join(mocap_folder, f'{audio_file.split(".")[0]}*.bvh'))
        # try to find the label
        if len(bvh_file) == 0:
            print(f'{audio_file} has no bvh file')
            continue
        label_file = glob.glob(os.path.join(label_folder, f'{audio_file.split(".")[0]}*.json'))
        if len(label_file) == 0:
            print(f'{audio_file} has no label file')
            continue
        audio_file = os.path.join(audio_folder, audio_file)
        audio_file = '/static/' + os.path.relpath(audio_file, static_folder)
        bvh_file = '/static/' + os.path.relpath(bvh_file[0], static_folder)
        label_file = '/static/' + os.path.relpath(label_file[0], static_folder)
        files_info.append({
            'audio': os.path.basename(audio_file), 
            'mocap': os.path.basename(bvh_file),
            'audio_path': audio_file, 
            'mocap_path': bvh_file, 
            'label_path': label_file})
        # URL encode the file names
    for i, file in enumerate(files_info):
        file['index'] = i
    return render_template('list_mocap_files.html', files=files_info)

def format_label(label):
    all_records = []
    for key, val in label.items():
        if key == 't_pose' or key == 'last':
            for (start, end) in val:
                all_records.append({
                    'start': start,
                    'end': end,
                    'label': key,
                    'action': key
                })
        elif '_yy_' in key:
            action, tag = key.split('_yy_')
            for (start, end) in val:
                all_records.append({
                    'start': start,
                    'end': end,
                    'label': tag,
                    'action': action
                })
    all_records.sort(key=lambda x: x['start'])
    return all_records

@app.route('/visualize/<int:index>')
def visualize(index):
    if len(files_info) == 0:
        # 重定向到列表页面
        return redirect(url_for('list_mocap_files'))
    # URL decode the file names
    mocap_bvh_name = urllib.parse.unquote(files_info[index]['mocap_path'])
    audio_name = urllib.parse.unquote(files_info[index]['audio_path'])
    label_name = urllib.parse.unquote(files_info[index]['label_path'])
    label = read_json_label(os.path.join(static_folder, label_name.replace('/static/', '')))
    label = format_label(label)
    return render_template('upload_keypoints3d.html', mocap_bvh_name=mocap_bvh_name, audio_name=audio_name, label_name=label_name, labels=label)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)