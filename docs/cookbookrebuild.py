import sys
import subprocess


def do_rebuild(app):
    subprocess.check_call([sys.executable, 'build.py'], cwd='..')


def setup(app):
    app.connect('builder-inited', do_rebuild)
    return {'version': '0.1'}
