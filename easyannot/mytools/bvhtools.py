import os
import numpy as np
import torch
import bvhtoolbox as bt

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    if len(rot_vecs.shape) > 2:
        rot_vec_ori = rot_vecs
        rot_vecs = rot_vecs.view(-1, 3)
    else:
        rot_vec_ori = None
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    if rot_vec_ori is not None:
        rot_mat = rot_mat.reshape(*rot_vec_ori.shape[:-1], 3, 3)
    return rot_mat

def read_bvh(file_path):
    with open(file_path, "r") as f:
        bvh = bt.BvhTree(f.read())
    return bvh

def get_offsets(bvh):
    joint_names = [joint.name for joint in bvh.get_joints()]
    offsets = []
    for joint_name in joint_names:
        offsets.append(bvh.joint_offset(joint_name))
    return offsets

def get_kintree(data):
    joints = data.get_joints()
    joint_names = data.get_joints_names()
    parents = []
    for joint in joints:
        if joint.name in ["Hips", "pelvis"]:
            print(f'Set {joint.name} as the root joint')
            parents.append(-1)
        else:
            parent_name = joint.parent.name
            parents.append(joint_names.index(parent_name))
    assert parents.count(-1) == 1, f'There should be only one root joint, but found {parents.count(-1)}'
    return parents

def rot_mat2trans_mat(rot_mat):
    trans_mat = torch.eye(4, device=rot_mat.device)
    trans_mat = torch.tile(trans_mat, (*rot_mat.shape[:-2], 1, 1))
    trans_mat[..., :3, :3] = rot_mat
    return trans_mat

def trans2trans_mat(trans):
    assert trans.shape[-1] == 3
    trans_mat = torch.eye(4, device=trans.device)
    trans_mat = torch.tile(trans_mat, (*trans.shape[:-1], 1, 1))
    trans_mat[..., :3, 3] = trans
    return trans_mat

def forward_kinematics(parents, offset, R):
    # the motions should use the same offsets
    assert len(R.shape) == 4  # T, J, 3, 3
    assert len(offset.shape) == 2  # J, 3
    R = rot_mat2trans_mat(R)
    offset = trans2trans_mat(offset)

    skeleton_trans = offset @ R
    results = [skeleton_trans[:, 0]]
    results.extend(
        results[parent_idx] @ skeleton_trans[:, idx]
        for idx, parent_idx in enumerate(parents)
        if parent_idx != -1
    )
    results = torch.stack(results, dim=0).permute(1, 0, 2, 3)
    return results[:, :, :3, 3]

class BVHSkeleton(torch.nn.Module):
    def __init__(self, bvh_path, scale=1.0):
        super().__init__()
        assert os.path.exists(bvh_path), f'BVH file {bvh_path} does not exist'
        self.bvh = read_bvh(bvh_path)
        self.joint_names = self.bvh.get_joints_names()
        print(f'>>> Read {len(self.joint_names)} joints: {self.joint_names}')
        # motion = np.array(self.bvh.frames, dtype=np.float32)
        offsets = get_offsets(self.bvh)
        offsets = torch.FloatTensor(offsets) * scale
        self.register_buffer('offsets', offsets)
        parents = get_kintree(self.bvh)
        self.parents = parents
    
    @property
    def num_joints(self):
        return len(self.joint_names)

    def init_params(self, num_frames=1):
        return {
            'offsets': self.offsets.clone(),
            'poses': torch.zeros(num_frames, self.num_joints, 3),
            'trans': torch.zeros(num_frames, 3),
        }
    
    def forward(self, params, fast_forward=False, input_rotation_matrix=False):
        offsets = params['offsets']
        poses = params['poses']
        batch_size, num_joints = poses.shape[0], poses.shape[1]
        if input_rotation_matrix:
            rot_mats = poses
        else:
            rot_mats = batch_rodrigues(
                poses.view(-1, 3)).view([batch_size, -1, 3, 3])
        positions = forward_kinematics(self.parents, offsets, rot_mats)
        trans = params['trans']
        positions = positions + trans[:, None, :]
        return {
            'keypoints3d': positions,
        }