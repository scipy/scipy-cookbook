import os
import sys
import subprocess


def do_rebuild(app):
    if not os.path.isfile(os.path.join('items', 'index.txt')):
        subprocess.check_call([sys.executable, 'build.py'], cwd='..')


def setup(app):
    app.connect('builder-inited', do_rebuild)
    return {'version': '0.1'}
