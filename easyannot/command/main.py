import os
help_text = ''' EasyAnnot is a simple tool for annotating data.
'''

def check_path(root, annot):
    if os.path.isabs(annot):
        return annot
    else:
        return os.path.abspath(os.path.join(root, annot))

def main_entrypoint():
    import argparse
    parser = argparse.ArgumentParser(usage=help_text)
    parser.add_argument('mode', type=str)
    parser.add_argument('root', type=str)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument('--annots', type=str, default='annots')
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--camera', type=str, default='')
    parser.add_argument('--points', type=str, default='points.json')
    parser.add_argument('--port', type=int, default=3456)
    parser.add_argument('--readonly', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.mode == 'camera':
        from .camera_viewer import app
        app.config['ROOT'] = os.path.abspath(args.root)
        app.config['CAMERA_ROOT'] = check_path(args.root, args.camera)
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'images':
        from .image_viewer import app
        app.config['ROOT'] = os.path.join(os.path.abspath(args.root), args.images)
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'annots':
        from .annot_viewer import app
        image_root = check_path(args.root, args.images)
        annot_root = os.path.join(check_path(args.root, args.annots), args.images)
        app.config['IMAGE_ROOT'] = image_root
        app.config['ANNOT_ROOT'] = annot_root
        print(f'Image root: {image_root}, Annot root: {annot_root}')
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'keypoints3d':
        from .keypoints_viewer import app
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'all':
        image_root = check_path(args.root, args.images)
        annot_root = os.path.join(check_path(args.root, args.annots), args.images)
        app.config['IMAGE_ROOT'] = image_root
    elif args.mode == 'match':
        from .annotate_match import app
        image_root = check_path(args.root, args.images)
        app.config['ROOT'] = os.path.abspath(args.root)
        app.config['CAMERA'] = args.camera
        app.config['IMAGE_ROOT'] = image_root
        app.config['POINTS_NAME'] = check_path(args.root, args.points)
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'match_group':
        from .annotate_match_group import app
        image_root = check_path(args.root, args.images)
        app.config['frame'] = args.frame
        app.config['ROOT'] = os.path.abspath(args.root)
        app.config['CAMERA'] = args.camera
        app.config['IMAGE_ROOT'] = image_root
        app.config['POINTS_NAME'] = check_path(args.root, args.points)
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'vanish':
        from .annotate_vanish import app
        image_root = check_path(args.root, args.images)
        app.config['ROOT'] = os.path.abspath(args.root)
        app.config['IMAGE_ROOT'] = image_root
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    else:
        print(f'Unknown mode: {args.mode}')
        raise NotImplementedError

if __name__ == '__main__':
    pass