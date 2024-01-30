from setuptools import setup, find_packages
import glob
import os

this_directory = os.path.abspath(os.path.dirname(__file__))

def read_file(filename):
    with open(os.path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

setup(
    name="easyannot",
    version="0.0.0",
    author="Qing Shuai",
    url="https://github.com/chingswy/EasyAnnot",
    author_email="s_q@zju.edu.cn",
    description="A simple tool for annotating data",
    long_description_content_type="text/markdown",
    long_description=open("Readme.md").read(),
    packages=find_packages(exclude=('examples', 'examples.*')),
    entry_points={
        'console_scripts': [
            'easyannot=easyannot.command.main:main_entrypoint',
        ],
    },
    install_requires=['flask'],
    package_data={
        'easyannot': ['static/*', 'templates/*']
    }
)
