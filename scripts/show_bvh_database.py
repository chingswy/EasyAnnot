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

print(f'template_folder: {template_folder}')
print(f'{os.listdir(template_folder)}')

app = Flask(__name__, 
            template_folder=template_folder, static_folder=static_folder)

# ├── audio
# │   ├── zhuge_c0013_i1_001.wav
# ├── MoCap_bvh
# │   ├── zhuge_c0013_i1_001_rx.bvh
# ├── MoCap_bvh
files_info = {}

def read_json_label(label_path):
    with open(label_path, 'r') as f:
        return json.load(f)

@app.route('/list_mocap_files/<day>')
def list_mocap_files(day):
    root_path = args.root
    mocap_folder = os.path.join(root_path, day, 'MoCap_bvh_align')
    if not os.path.exists(mocap_folder):
        mocap_folder = os.path.join(root_path, day, 'MoCap_bvh')
    label_folder = os.path.join(root_path, day, 'label')
    audio_folder = os.path.join(root_path, day, 'audio')

    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
    files_info[day] = []
    for audio_file in audio_files:
        # try to find the bvh file
        bvh_file = glob.glob(os.path.join(mocap_folder, f'{audio_file.split(".")[0]}*.bvh'))
        # try to find the label
        if len(bvh_file) == 0:
            print(f'{audio_file} in {mocap_folder} has no bvh file')
            continue
        audio_file = os.path.join(audio_folder, audio_file)
        audio_file = '/static/' + os.path.relpath(audio_file, static_folder)
        bvh_file = '/static/' + os.path.relpath(bvh_file[0], static_folder)
        label_file = glob.glob(os.path.join(label_folder, f'{audio_file.split(".")[0]}*.json'))
        if len(label_file) == 0:
            # print(f'{audio_file} in {label_folder} has no label file')
            label_file = ''
        else:
            label_file = '/static/' + os.path.relpath(label_file[0], static_folder)
        files_info[day].append({
            'audio': os.path.basename(audio_file), 
            'mocap': os.path.basename(bvh_file),
            'audio_path': audio_file, 
            'mocap_path': bvh_file, 
            'label_path': label_file})
        # URL encode the file names
    for i, file in enumerate(files_info[day]):
        file['index'] = i
    return render_template('list_mocap_files.html', files=files_info[day], day=day)

@app.route('/')
def list_days():
    root_path = args.root
    # Get all subdirectories in root path
    days = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # Sort the directories
    days.sort()
    return render_template('index_any_folder.html', href='list_mocap_files', days=days)

def format_label(label):
    # [{'start_pose': 't_pose', 'end_pose': 't_pose', 'semantic_label': '', 'start_time': 0, 'end_time': 2.81906}, {'start_pose': 'i2', 'end_pose': 'i2', 'semantic_label': '', 'start_time': 2.819, 'end_time': 94.34608}, {'start_pose': 'last', 'end_pose': 'last', 'semantic_label': '', 'start_time': 94.346, 'end_time': 94.533}]
    label.sort(key=lambda x: x['start_time'])
    return label

@app.route('/visualize/<day>/<int:index>')
def visualize(day, index):
    if len(files_info[day]) == 0:
        # 重定向到列表页面
        return redirect(url_for('list_mocap_files', day=day))
    # URL decode the file names
    mocap_bvh_name = urllib.parse.unquote(files_info[day][index]['mocap_path'])
    audio_name = urllib.parse.unquote(files_info[day][index]['audio_path'])
    label_name = urllib.parse.unquote(files_info[day][index]['label_path'])
    if label_name == '':
        label = []
    else:
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