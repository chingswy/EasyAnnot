import os
help_text = ''' EasyAnnot is a simple tool for annotating data.
'''

def main_entrypoint():
    import argparse
    parser = argparse.ArgumentParser(usage=help_text)
    parser.add_argument('mode', type=str)
    parser.add_argument('root', type=str)
    parser.add_argument('--images', type=str, default='images')
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

if __name__ == '__main__':
    pass