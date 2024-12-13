import os
import glob
from os.path import join
import torch
from flask import Flask, render_template, jsonify, send_file
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

skeleton_def_30 = {
    'joint_names': ['Hips', 'Chest', 'Chest2', 'Neck', 'Head', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'LeftFinger0', 'LeftFinger01', 'LeftFinger02', 'LeftFinger1', 'LeftFinger11', 'LeftFinger12', 'LeftFinger2', 'LeftFinger21', 'LeftFinger22', 'LeftFinger3', 'LeftFinger31', 'LeftFinger32', 'LeftFinger4', 'LeftFinger41', 'LeftFinger42', 'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist', 'RightFinger0', 'RightFinger01', 'RightFinger02', 'RightFinger1', 'RightFinger11', 'RightFinger12', 'RightFinger2', 'RightFinger21', 'RightFinger22', 'RightFinger3', 'RightFinger31', 'RightFinger32', 'RightFinger4', 'RightFinger41', 'RightFinger42', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe'],
}
skeleton_def_30['body_index'] = [skeleton_def_30['joint_names'].index(name) for name in \
    ['Hips', 'Chest', 'Chest2', 'Neck', 'Head', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightCollar', 'RightShoulder', 'RightElbow', 'RightWrist', 'LeftHip', 'LeftKnee', 'LeftAnkle', 'LeftToe', 'RightHip', 'RightKnee', 'RightAnkle', 'RightToe']]
skeleton_def_30['body_index_wo_wrist'] = [skeleton_def_30['joint_names'].index(name) for name in \
    ['Hips', 'Chest', 'Chest2', 'Neck', 'Head', 'LeftCollar', 'LeftShoulder', 'LeftElbow', 'RightCollar', 'RightShoulder', 'RightElbow', 'LeftHip', 'LeftKnee', 'RightHip', 'RightKnee']]

skeleton_def_31 = {
    'joint_names': ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase'],
}
skeleton_def_31['body_index'] = [skeleton_def_31['joint_names'].index(name) for name in \
    ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot']]
skeleton_def_31['body_index_wo_wrist'] = [skeleton_def_31['joint_names'].index(name) for name in \
    ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'LeftShoulder', 'LeftArm', 'RightShoulder', 'RightArm', 'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg']]

def get_body_index(joint_names):
    # check the skeleton definition
    if 'LeftHandThumb1' in joint_names:
        return skeleton_def_31['body_index'], skeleton_def_31['body_index_wo_wrist']
    else:
        return skeleton_def_30['body_index'], skeleton_def_30['body_index_wo_wrist']

def clips_from_flags(flags):
    """
       flags: np.ndarray(n, )
       return: list of (start, end)
    """
    clips = []
    start = None
    
    # 遍历flags数组，找到区间的开始和结束
    for i, flag in enumerate(flags):
        if flag and start is None:
            # 如果当前标志非零且还没有记录起始点，则记录起始点
            start = i
        elif not flag and start is not None:
            # 如果当前标志为零且已经记录了起始点，则记录结束点，并添加区间到结果列表
            end = i
            clips.append((start, end))
            start = None  # 重置起始点，准备下一个区间
    
    # 如果flags以非零值结束，需要添加最后一个区间
    if start is not None:
        clips.append((start, len(flags)))
    
    return clips

from easyannot.mytools.bvhtools import BVHSkeleton, read_bvh

def get_joint_positions(bvh_file_path):
    from scipy.spatial.transform import Rotation as R
    bvh_data = BVHSkeleton(bvh_file_path, verbose=False)
    joint_names = bvh_data.bvh.get_joints_names()
    body_index, body_index_wo_wrist = get_body_index(joint_names)
    # frames: (nframes, (njoints+1)*3)
    frames = np.array(bvh_data.bvh.frames, dtype=np.float32)
    angles = frames[:, 3:].reshape(frames.shape[0], -1, 3)
    # 从欧拉角转旋转矩阵
    angles_flat = angles.reshape(-1, 3)
    rotation = R.from_euler("ZXY", angles_flat, degrees=True).as_matrix().reshape(angles.shape[0], -1, 3, 3)
    # 从旋转矩阵转关节位置
    params = {
        'poses': torch.FloatTensor(rotation),
        'offsets': bvh_data.offsets,
        'trans': torch.FloatTensor(frames[:, :3])
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key, val in params.items():
        params[key] = val.to(device)
    positions = bvh_data.forward(params, input_rotation_matrix=True)['keypoints3d']
    angles_body = angles[:, body_index_wo_wrist]
    return angles_body, angles, rotation, positions.cpu().numpy(), positions[:, body_index].cpu().numpy()

@app.route('/list_mocap_files/<day>')
def list_mocap_files(day):
    root_path = args.root
    mocap_folder = os.path.join(root_path, day, args.mocap_dir)
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
        # audio_file = os.path.relpath(audio_file, root_path)
        audio_file = os.path.basename(audio_file)
        # audio_file = '/static/' + os.path.relpath(audio_file, static_folder)
        # bvh_file = '/static/' + os.path.relpath(bvh_file[0], static_folder)
        bvh_file = bvh_file[0]
        bvh_file = os.path.basename(bvh_file)
        files_info[day].append({
            'audio': os.path.basename(audio_file), 
            'mocap': os.path.basename(bvh_file),
            'audio_path': audio_file, 
            'mocap_path': bvh_file, 
            })
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

def _get_audio_energy(audio_file_path):
    import whisper
    assert os.path.exists(audio_file_path), f"{audio_file_path} not exists"
    audio = whisper.load_audio(audio_file_path)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128)
    mel = mel.T
    # (128, 100 * seconds)
    # target_length = int(mel.shape[0] * target_fps / source_fps)
    # mel = torch.nn.functional.interpolate(mel.T[None], size=target_length, mode='linear', align_corners=False)[0].T
    mel_energy_raw = torch.norm(torch.exp(mel), dim=-1, keepdim=True)
    # 归一化mel_energy
    trim_start, trim_end = 50, 50
    mel_energy = (mel_energy_raw - mel_energy_raw[trim_start:-trim_end].min()) / (mel_energy_raw[trim_start:-trim_end].max() - mel_energy_raw[trim_start:-trim_end].min())
    return mel_energy

@app.route('/get_audio_file/<day>/<filename>')
def get_audio_file(day, filename):
    root_path = args.root
    print(f'get audio file: {day} -> {filename}')
    audio_file_path = os.path.join(root_path, day, 'audio', filename)
    if os.path.exists(audio_file_path):
        return send_file(audio_file_path)
    else:
        import ipdb; ipdb.set_trace()

@app.route('/get_bvh_file/<day>/<filename>')
def get_bvh_file(day, filename):
    root_path = args.root
    print(f'get bvh file: {day} -> {filename}')
    bvh_file_path = os.path.join(root_path, day, args.mocap_dir, filename)
    if os.path.exists(bvh_file_path):
        return send_file(bvh_file_path)
    else:
        import ipdb; ipdb.set_trace()

def _update_flag_by_clips(flag, min_frame):
    clips = clips_from_flags(flag)
    flag_update = np.zeros_like(flag)
    for clip in clips:
        if clip[1] - clip[0] > min_frame:
            flag_update[clip[0]:clip[1]] = True
    return flag_update

# 判断条件
# 选择能用的片段的标准：
# 对于一个能用的片段中的每一帧，要求
# 1. 不是 T Pose
# 2. 大小为S的窗口内的平均音频能量 > 阈值
# 3. 大小为S窗口内的平均速度 > 阈值
# motion条件导致的问题：部分区间有音频，但是姿态是保持静止的
# audio条件导致的问题：话说完了，然后过渡回rest pose

# 对于一个获得的片段，进一步：
# 1. 起始帧的速度 < 阈值，如果不是，需要向右收缩，直到找到阈值
# 2. 结束帧的速度 < 阈���，如果不是，需要向后延长，直到找到阈值


def _find_main_clips(angles_body, audio_energy, positions_body,
                     tpose_angle_thres=20,
                     audio_energy_thres=0.15,
                     min_vel_threshold = 0.1,
                     motion_fps = 120
                     ):
    """
        用于寻找 有效区间
    """
    min_silent_frame = args.silent_frame
    # calculate motion velocity
    vel = np.linalg.norm(positions_body[1:] - positions_body[:-1], axis=-1).max(axis=-1)
    vel = np.insert(vel, 0, vel[0])
    # 1. 排除无效区间
    angle_max = np.abs(angles_body).max(axis=-1).max(axis=-1)
    flag_is_tpose = angle_max < tpose_angle_thres # degree
    flag_is_tpose = _update_flag_by_clips(flag_is_tpose, min_frame=5)
    # check audio
    # smooth audio energy
    window_size = 5
    # padding window_size
    audio_energy = np.pad(audio_energy, (window_size, window_size), mode='edge')
    audio_energy = np.convolve(audio_energy, np.ones(2*window_size+1), mode='valid') / (2*window_size+1)
    flag_no_audio = _update_flag_by_clips(audio_energy < audio_energy_thres, min_frame=min_silent_frame)
    # 组合flag
    flag_main_clip = ~flag_is_tpose
    flag_main_clip = flag_main_clip & (~flag_no_audio)
    # flag_main_clip = flag_main_clip & (~flag_no_motion)
    clips_main_clip = clips_from_flags(flag_main_clip)
    labels = []
    min_window_size = 120
    for (start, end) in clips_main_clip:
        if end - start < min_window_size:
            continue
        # 起始帧的速度 < 阈值
        # 找一个最近的start，使得vel[start] < 0.1，同时往左右两边扩展
        # Adjust the start and end indices of the clip to ensure the velocity is below the threshold.
        # If the velocity at the start is above the threshold, search for the nearest point within the clip
        # where the velocity is below the threshold, and adjust the start index accordingly.
        if vel[start] > min_vel_threshold:
            for offset in range(0, end - start):
                if vel[start + offset] < min_vel_threshold and \
                    not flag_is_tpose[start + offset] and \
                        flag_no_audio[start + offset]:
                    start = start + offset
                    break
                elif start - offset >= 0 and \
                    vel[start - offset] < min_vel_threshold and \
                        not flag_is_tpose[start - offset] and \
                            flag_no_audio[start - offset]:
                    start = start - offset
                    break
            else:
                start = end
                continue
        # Similarly, adjust the end index if the velocity at the end is above the threshold.
        if vel[end-1] > min_vel_threshold:
            for offset in range(1, end - start):
                if vel[end - offset] < min_vel_threshold and \
                    not flag_is_tpose[end - offset] and \
                        flag_no_audio[end - offset]:
                    end = end - offset
                    break
                elif end + offset < vel.shape[0] and \
                    vel[end + offset] < min_vel_threshold and \
                        not flag_is_tpose[end + offset] and \
                            flag_no_audio[end + offset]:
                    end = end + offset
                    break
            else:
                end = start
                continue
        if end - start < min_window_size:
            continue
        labels.append({
            'category': 'main',
            'label': '',
            'title': '',
            'start_frame': start,
            'end_frame': end,
            'start_time': start / motion_fps,
            'end_time': end / motion_fps
        })
    return labels, vel

def _get_labels(audio_name, bvh_name, strict=False):
    # audio_energy: (second x 100, 1)
    audio_energy = _get_audio_energy(audio_name)
    audio_fps, motion_fps = 100, 120
    target_length = int(audio_energy.shape[0] * motion_fps / audio_fps)
    audio_energy = torch.nn.functional.interpolate(audio_energy.T[None], size=target_length, mode='linear', align_corners=False)[0].T
    # motion
    angles_body, angles, rotations, positions, positions_body = get_joint_positions(bvh_name)
    if strict:
        # check the length of audio and motion
        if abs(audio_energy.shape[0] - positions.shape[0]) > 30:
            print(f'mismatch: {os.sep.join(audio_name.split(os.sep)[-3:]):40s}: {audio_energy.shape[0]/motion_fps:7.3f} - {positions.shape[0]/motion_fps:7.3f} = {abs(audio_energy.shape[0] - positions.shape[0])/motion_fps:5.3f}')
            return None, None, None
    if audio_energy.shape[0] > positions.shape[0]:
        audio_energy = audio_energy[:positions.shape[0]]
    else:
        # padding audio_energy
        padding = positions.shape[0] - audio_energy.shape[0]
        audio_energy = torch.cat([audio_energy, audio_energy[-1:].repeat(padding, 1)])
    audio_energy = audio_energy[:, 0].numpy()
    labels_main, vel = _find_main_clips(angles_body, audio_energy, positions_body)
    return labels_main, audio_energy, vel

@app.route('/get_labels/<day>/<audio_name>/<bvh_name>')
def get_labels(day, audio_name, bvh_name, strict=False):
    root_path = args.root
    label_file_path = os.path.join(root_path, day, args.label_dir, audio_name.replace('.wav', '.json'))
    audio_file_path = os.path.join(root_path, day, 'audio', audio_name)
    bvh_file_path = os.path.join(root_path, day, args.mocap_dir, bvh_name)
    labels_main, audio_energy, vel = _get_labels(audio_file_path, bvh_file_path, strict)
    # 寻找有效的区间
    if not os.path.exists(label_file_path):
        # 读入bvh文件与audio文件
        # 计算rotation与单位阵的差异
        labels = labels_main
    else:
        labels = read_json_label(label_file_path)
        if len([l for l in labels if l['category'] == 'main']) == 0:
            labels.extend(labels_main)
    ret = {
        'audio_energy': audio_energy.tolist(),
        'motion_velocity': vel.tolist(),
        'labels': labels
    }
    return jsonify(ret)

@app.route('/visualize/<day>/<int:index>')
def visualize(day, index):
    if day not in files_info or len(files_info[day]) == 0:
        # 重定向到列表页面
        return redirect(url_for('list_mocap_files', day=day))
    # URL decode the file names
    mocap_bvh_name = urllib.parse.unquote(files_info[day][index]['mocap_path'])
    print(f'mocap_bvh_name: {mocap_bvh_name}')
    audio_name = urllib.parse.unquote(files_info[day][index]['audio_path'])
    prev_index = index - 1 if index > 0 else None
    next_index = index + 1 if index < len(files_info[day]) - 1 else None

    prev_url = url_for('visualize', day=day, index=prev_index) if prev_index is not None else None
    next_url = url_for('visualize', day=day, index=next_index) if next_index is not None else None

    return render_template('annot_bvh.html', day=day, mocap_bvh_name=mocap_bvh_name, audio_name=audio_name,
                           prev_url=prev_url, next_url=next_url)

def generate_labels(args):
    from tqdm import tqdm
    # Get all subdirectories in root path
    root_path = args.root
    days = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    # Sort the directories
    days.sort()
    for day in days:
        print(f'generate labels for {day}')
        mocap_folder = os.path.join(root_path, day, args.mocap_dir)
        audio_folder = os.path.join(root_path, day, 'audio')

        audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.wav')])
        files_info[day] = []
        for audio_file in tqdm(audio_files, desc=f'{day}'):
            label_file_name = os.path.join(root_path, day, args.label_dir, os.path.basename(audio_file).replace('.wav', '.json'))
            # try to find the bvh file
            bvh_file = glob.glob(os.path.join(mocap_folder, f'{audio_file.split(".")[0]}*.bvh'))
            # try to find the label
            if len(bvh_file) == 0:
                print(f'{audio_file} in {mocap_folder} has no bvh file')
                continue
            bvh_file_path = bvh_file[0]

            if os.path.exists(label_file_name):
                # read the label
                with open(label_file_name, 'r') as f:
                    labels = json.load(f)
                # 检查一下total frames
                # bvh = read_bvh(bvh_file_path)
                # total_frames = len(bvh.frames)
                if len(labels) == 0:
                    total_frames = 0
                else:
                    total_frames = labels[-1]['end_frame'] - labels[0]['start_frame']
                files_info[day].append({
                    'total_frames': total_frames,
                    'valid_frames': sum([l['end_frame'] - l['start_frame'] for l in labels if l['category'] == 'main']),
                    'valid_clips': len([l for l in labels if l['category'] == 'main']),
                })
                continue
            os.makedirs(os.path.dirname(label_file_name), exist_ok=True)

            audio_file = os.path.join(audio_folder, audio_file)
            labels_main, audio_energy, vel = _get_labels(audio_file, bvh_file[0], strict=True)
            if labels_main is None:
                # write 一个空的
                with open(label_file_name, 'w') as f:
                    json.dump([], f, indent=4)
                continue
            files_info[day].append({
                'total_frames': vel.shape[0],
                'valid_frames': sum([l['end_frame'] - l['start_frame'] for l in labels_main]),
                'valid_clips': len(labels_main),
            })
            with open(label_file_name, 'w') as f:
                json.dump(labels_main, f, indent=4)
    # 写入report文件
    with open(os.path.join('./', 'report_clips.json'), 'w') as f:
        json.dump(files_info, f, indent=4)
    # 分别统计每一天的量，形成���格输出
    headers = ['day', 'total_frames', 'valid_frames', 'valid_clips']
    rows = []
    total_frames_sum = 0
    valid_frames_sum = 0
    valid_clips_sum = 0

    for day, info in files_info.items():
        total_frames = sum([l['total_frames'] for l in info])
        valid_frames = sum([l['valid_frames'] for l in info])
        valid_clips = sum([l['valid_clips'] for l in info])
        rows.append([day, total_frames, valid_frames, valid_clips])

        # Accumulate totals
        total_frames_sum += total_frames
        valid_frames_sum += valid_frames
        valid_clips_sum += valid_clips

    # Add a summary row
    rows.append(['Total', total_frames_sum, valid_frames_sum, valid_clips_sum])

    from tabulate import tabulate
    print(tabulate(rows, headers=headers, tablefmt='grid'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--mocap_dir', type=str, default='MoCap_bvh_align')
    parser.add_argument('--label_dir', type=str, default='label_trim')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--silent_frame', type=int, default=30)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()

    args.root = os.path.abspath(args.root)

    if args.generate:
        generate_labels(args)
    else:
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)

    # python3 -m pip install openai-whisper
    # python3 setup.py develop
    # python3 scripts/annotate_bvh_database.py --root /apdcephfs_cq10/share_1467498/datasets/motion_data/a2g/project --mocap_dir MoCap_bvh --port 5001 --silent_frame 120
    # failure case
    # http://9.134.230.186:5001/visualize/20240612_ZGL/46
    # python3 scripts/annotate_bvh_database.py --root /apdcephfs_cq10/share_1467498/datasets/motion_data/a2g/project --mocap_dir MoCap_bvh --port 5001 --silent_frame 240 --generate