import os
import yaml

def read_yaml(filename):
    if not os.path.exists(filename):
        if input(f'File {filename} not found, do you want to create a new one? (y/n)') == 'y':
            write_yaml(filename, {})
        else:
            return 0
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def write_yaml(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        yaml.dump(data, f)