# 这个脚本用于读取相机参数，接着将文件夹中的所有图片进行去畸变处理，最后保存到新的文件夹中
import os
from tqdm import tqdm
from os.path import join
import numpy as np
import cv2

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.6f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == 'int':
            self._write('{}: {}'.format(key, value))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        elif dt == 'int':
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        H = intri.read('H_{}'.format(cam), dt='int')
        W = intri.read('W_{}'.format(cam), dt='int')
        if H is None or W is None:
            print('[camera] no H or W for {}'.format(cam))
            H, W = -1, -1
        cams[cam]['H'] = H
        cams[cam]['W'] = W
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        assert Rvec is not None, cam
        R = cv2.Rodrigues(Rvec)[0]
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - R.T @ Tvec
        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
        if cams[cam]['dist'] is None:
            cams[cam]['dist'] = intri.read('D_{}'.format(cam))
            if cams[cam]['dist'] is None:
                print('[camera] no dist for {}'.format(cam))
    return cams


def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'H' in val.keys() and 'W' in val.keys():
            intri.write('H_{}'.format(key), val['H'], dt='int')
            intri.write('W_{}'.format(key), val['W'], dt='int')
        assert val['R'].shape == (3, 3), f"{val['R'].shape} must == (3, 3)"
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])

def map_folders(src, dst, mapx, mapy):
    srcnames = sorted(os.listdir(src))
    for srcname in tqdm(srcnames):
        dstname = join(dst, srcname)
        if os.path.exists(dstname):
            continue
        img = cv2.imread(join(src, srcname))
        img_dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        os.makedirs(dst, exist_ok=True)
        cv2.imwrite(dstname, img_dst)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    subs = sorted(os.listdir(os.path.join(args.path, 'images')))
    cameras = read_camera(os.path.join(args.path, 'intri.yml'), os.path.join(args.path, 'extri.yml'))
    cameras_new = {}
    for sub in subs:
        camera = cameras[sub]
        width, height = camera['W'], camera['H']
        newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'], 
                            (width, height), 0, (width,height), centerPrincipalPoint=True)
        mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
        map_folders(join(args.path, 'images', sub), join(args.path, 'images_undis', sub), mapx, mapy)
        cameras_new[sub] = camera.copy()
        cameras_new[sub]['K'] = newK
        cameras_new[sub]['dist'] = np.zeros_like(camera['dist'])
    write_camera(cameras_new, join(args.path, 'cameras_undis'))