import numpy as np
import cv2

def batch_triangulate(keypoints_, Pall, min_view=2):
    """ triangulate the keypoints of whole body

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4): projection matrix of each view
        min_view (int, optional): min view for visible points. Defaults to 2.

    Returns:
        keypoints3d: (nJoints, 4)
    """
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d #* (conf[..., 0].sum(axis=-1)>min_view)
    return result

def project_wo_dist(keypoints, RT, einsum='vab,kb->vka'):
    homo = np.concatenate([keypoints[..., :3], np.ones_like(keypoints[..., :1])], axis=-1)
    kpts2d = np.einsum(einsum, RT, homo)
    depth = kpts2d[..., 2]
    kpts2d[..., :2] /= kpts2d[..., 2:]
    return kpts2d, depth

def project_w_dist(k3d, camera):
    k3d0 = np.ascontiguousarray(k3d[:, :3])
    k3d_rt = np.dot(k3d0, camera['R'].T) + camera['T'].T
    depth = k3d_rt[:, -1:]
    k2d, _ = cv2.projectPoints(k3d0, camera['R'], camera['T'], camera['K'], camera['dist'])
    k2d = np.hstack([k2d[:, 0], k3d[:, -1:]])
    return k2d, depth