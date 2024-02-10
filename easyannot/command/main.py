import os
help_text = ''' EasyAnnot is a simple tool for annotating data.
'''

def main_entrypoint():
    import argparse
    parser = argparse.ArgumentParser(usage=help_text)
    parser.add_argument('mode', type=str)
    parser.add_argument('root', type=str)
    parser.add_argument('--images', type=str, default='images')
    parser.add_argument('--annots', type=str, default='annots')
    parser.add_argument('--port', type=int, default=3456)
    parser.add_argument('--readonly', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.mode == 'camera':
        from .camera_viewer import app
        app.config['ROOT'] = os.path.abspath(args.root)
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'images':
        from .image_viewer import app
        app.config['ROOT'] = os.path.abspath(args.root)
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'annots':
        from .annot_viewer import app
        image_root = os.path.join(os.path.abspath(args.root), args.images)
        annot_root = os.path.abspath(args.annots)
        app.config['IMAGE_ROOT'] = image_root
        app.config['ANNOT_ROOT'] = annot_root
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    elif args.mode == 'keypoints3d':
        from .keypoints_viewer import app
        app.run(debug=args.debug, port=args.port, host='0.0.0.0')
    else:
        raise NotImplementedError

if __name__ == '__main__':
    pass