import os
from os.path import join
from easymocap.mytools.file_utils import read_json
from easymocap.mytools.camera_utils import read_cameras, write_camera

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(usage='''
    python3 apps/preprocess/extract_roi.py ${data} ${data}/../../../3150_4100_crop --images images_undis --cameras ${data}/cameras_undis
''')
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument('--cameras', type=str, default=None)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cropfile = join(args.path, 'roi.json')
    if args.cameras is None:
        cameras  = read_cameras(args.path)
    else:
        cameras  = read_cameras(args.cameras)
    rois = read_json(cropfile)
    for sub, camera in cameras.items():
        l, t, r, b = rois[sub]
        camera['K'][0, 2] -= l
        camera['K'][1, 2] -= t
        camera['H'] = b - t
        camera['W'] = r - l
    subs = sorted(os.listdir(join(args.path, args.images)))
    write_camera(cameras, args.out)    
    for sub in subs:
        l, t, r, b = rois[sub]
        srcpath = join(args.path, args.images, sub, '%06d.jpg')
        dstpath = join(args.out, 'images', sub)
        os.makedirs(dstpath, exist_ok=True)
        len_dst = len(os.listdir(dstpath))
        len_src = len(os.listdir(join(args.path, args.images, sub)))
        if len_dst == len_src:
            continue
        cmd = f'ffmpeg -start_number 0 -i {srcpath} -filter:v "crop={r-l}:{b-t}:{l}:{t}" -start_number 0 -c:a copy {dstpath}/%06d.jpg'
        print(cmd)
        os.system(cmd)
