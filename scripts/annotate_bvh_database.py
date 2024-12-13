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
        if args.check_clips:
            clip_file = os.path.join(root_path, day, args.label_dir, audio_file.replace('.wav', '.json'))
            if os.path.exists(clip_file):
                clip_info = read_json_label(clip_file)
                clip_main = [l for l in clip_info if l['category'] == 'main']
                if len(clip_main) > 0:
                    continue
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
    days_valid = []
    for day in days:
        audio_dir = os.path.join(root_path, day, 'audio')
        mocap_dir = os.path.join(root_path, day, args.mocap_dir)
        if not os.path.exists(audio_dir) or not os.path.exists(mocap_dir):
            print(f'{day} has no audio or mocap')
            continue
        days_valid.append(day)
    return render_template('index_any_folder.html', href='list_mocap_files', days=days_valid)

def format_label(label):
    # [{'start_pose': 't_pose', 'end_pose': 't_pose', 'semantic_label': '', 'start_time': 0, 'end_time': 2.81906}, {'start_pose': 'i2', 'end_pose': 'i2', 'semantic_label': '', 'start_time': 2.819, 'end_time': 94.34608}, {'start_pose': 'last', 'end_pose': 'last', 'semantic_label': '', 'start_time': 94.346, 'end_time': 94.533}]
    label.sort(key=lambda x: x['start_time'])
    return label

def _normalize_energy(energy, trim_start=50, trim_end=50):
    return np.clip((energy - energy[trim_start:-trim_end].min()) / (energy[trim_start:-trim_end].max() - energy[trim_start:-trim_end].min()), 0, 1)

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
    audio_fps, motion_fps = 100, 120
    target_length = int(mel_energy_raw.shape[0] * motion_fps / audio_fps)
    # 这个插值会改变数据的大小，原理是通过线性插值将数据调整到目标长度
    mel_energy_raw = torch.nn.functional.interpolate(mel_energy_raw.T[None], size=target_length, mode='linear', align_corners=False)[0].T
    mel_energy_raw = mel_energy_raw.numpy()
    # smooth audio energy
    window_size = 5
    # padding window_size
    mel_energy_raw = np.pad(mel_energy_raw[:, 0], (window_size, window_size), mode='edge')
    mel_energy_raw = np.convolve(mel_energy_raw, np.ones(2*window_size+1), mode='valid') / (2*window_size+1)
    # normalize mel_energy_raw
    mel_energy_raw = _normalize_energy(mel_energy_raw, trim_start=50, trim_end=50)
    return mel_energy_raw

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
    # 速度需要进行窗口插值
    motion_window_size = 5
    vel = np.pad(vel, (motion_window_size, motion_window_size), mode='edge')
    vel = np.convolve(vel, np.ones(2*motion_window_size+1), mode='valid') / (2*motion_window_size+1)

    # 1. 排除无效区间
    angle_max = np.abs(angles_body).max(axis=-1).max(axis=-1)
    flag_is_tpose = angle_max < tpose_angle_thres # degree
    flag_is_tpose = _update_flag_by_clips(flag_is_tpose, min_frame=5)
    # check audio
    flag_no_audio_single_frame = audio_energy < audio_energy_thres
    flag_no_audio = _update_flag_by_clips(flag_no_audio_single_frame, min_frame=min_silent_frame)
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
        def _check_frame(frame_index):
            if frame_index < 0 or frame_index >= vel.shape[0]:
                return False
            flag_vel = vel[frame_index] < min_vel_threshold
            flag_tpose = flag_is_tpose[frame_index]
            flag_no_audio = flag_no_audio_single_frame[frame_index]
            return flag_vel and not flag_tpose and flag_no_audio

        if vel[start] > min_vel_threshold:
            step_size = 2
            print(f'  check start vel: {vel[start]} with step size = {step_size}')
            flag_found_peak = False
            for offset in range(step_size, end - start, step_size):
                if _check_frame(start - offset):
                    start = start - offset
                    flag_found_peak = True
                    break
                if _check_frame(start + offset):
                    start = start + offset
                    flag_found_peak = True
                    break
                if start - offset >= 0:
                    print(f'    {start - offset:6d} not satisfy: vel={vel[start - offset]:.3f}, is_tpose={flag_is_tpose[start - offset]}, no_audio={flag_no_audio_single_frame[start - offset]}')
                print(f'    {start + offset:6d} not satisfy: vel={vel[start + offset]:.3f}, is_tpose={flag_is_tpose[start + offset]}, no_audio={flag_no_audio_single_frame[start + offset]}')
            if not flag_found_peak:
                start = end
                print(f'  not found peak, set start = {start}')
                continue
            else:
                print(f'  found peak at {start}')
        # Similarly, adjust the end index if the velocity at the end is above the threshold.
        if vel[end-1] > min_vel_threshold:
            step_size = 2
            print(f'  check end vel: {vel[end-1]} with step size = {step_size}')
            flag_found_peak = False
            for offset in range(step_size, end - start, step_size):
                if _check_frame(end - offset):
                    end = end - offset
                    flag_found_peak = True
                    break
                if _check_frame(end + offset):
                    end = end + offset
                    flag_found_peak = True
                    break
                print(f'    {end - offset:6d} not satisfy: vel={vel[end - offset]:.3f}, is_tpose={flag_is_tpose[end - offset]}, no_audio={flag_no_audio_single_frame[end - offset]}')
                if end + offset < vel.shape[0]:
                    print(f'    {end + offset:6d} not satisfy: vel={vel[end + offset]:.3f}, is_tpose={flag_is_tpose[end + offset]}, no_audio={flag_no_audio_single_frame[end + offset]}')
            if not flag_found_peak:
                end = start
                print(f'  not found peak, set end = {end}')
                continue
            else:
                print(f'  found peak at {end}')
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
    if len(labels) == 0:
        # 打印debug信息：
        clip_audios = clips_from_flags(~flag_no_audio)
        print(f'clips have audios: {clip_audios}')
        clip_tpose = clips_from_flags(flag_is_tpose)
        print(f'clips have tpose: {clip_tpose}')
        print(f'clips main clip: {clips_main_clip}')
    return labels, vel

def _get_labels(audio_name, bvh_name, strict=False, motion_fps=120):
    # audio_energy: (second x 100, 1)
    audio_energy = _get_audio_energy(audio_name)
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
        audio_energy = np.concatenate([audio_energy, audio_energy[-1:].repeat(padding, axis=0)])
    audio_energy = audio_energy
    labels_main, vel = _find_main_clips(angles_body, audio_energy, positions_body, min_vel_threshold=args.min_vel_threshold)
    # 添加一个dummy的
    labels_main.append({
        'category': 'dummy',
        'label': '',
        'title': '',
        'start_frame': 0,
        'end_frame': vel.shape[0],
    })
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
        if not os.path.exists(audio_folder):
            print(f'{audio_folder} not exists')
            continue
        if not os.path.exists(mocap_folder):
            print(f'{mocap_folder} not exists')
            continue
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

            if os.path.exists(label_file_name) and not args.restart:
                # read the label
                with open(label_file_name, 'r') as f:
                    labels = json.load(f)
                # 检查一下total frames
                label_dummy = [l for l in labels if l['category'] == 'dummy']
                if len(label_dummy) == 0:
                    total_frames = 0
                else:
                    total_frames = label_dummy[0]['end_frame'] - label_dummy[0]['start_frame']
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
    # 分别统计每一天的量，形成格输出
    headers = ['day', 'total_frames', 'valid_frames', 'valid_ratio', 'valid_time', 'valid_clips']
    rows = []
    total_frames_sum = 0
    valid_frames_sum = 0
    valid_clips_sum = 0

    for day, info in files_info.items():
        total_frames = sum([l['total_frames'] for l in info])
        valid_frames = sum([l['valid_frames'] for l in info])
        valid_clips = sum([l['valid_clips'] for l in info])
        valid_ratio = valid_frames / total_frames
        valid_time = valid_frames / 120
        rows.append([day, total_frames, valid_frames, valid_ratio, valid_time, valid_clips])

        # Accumulate totals
        total_frames_sum += total_frames
        valid_frames_sum += valid_frames
        valid_clips_sum += valid_clips

    # Add a summary row
    rows.append(['Total', total_frames_sum, valid_frames_sum, valid_frames_sum / total_frames_sum, valid_frames_sum / 120, valid_clips_sum])

    from tabulate import tabulate
    print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))
    print(tabulate(rows, headers=headers, tablefmt='fancy_grid'), file=open('./report_clips_axx.txt', 'a'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--mocap_dir', type=str, default='MoCap_bvh_align')
    parser.add_argument('--label_dir', type=str, default='label_trim')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--silent_frame', type=int, default=30)
    parser.add_argument('--min_vel_threshold', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--check_clips', action='store_true')
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
    # 处理project_extend数据
    # python3 scripts/annotate_bvh_database.py --root /apdcephfs_cq10/share_1467498/datasets/motion_data/a2g/project_extend --mocap_dir MoCap_bvh_align --port 5001 --silent_frame 120 --min_vel_threshold 0.15 --generate