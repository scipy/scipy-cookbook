#!/usr/bin/env python
import os
import sys
import subprocess

if sys.argv[1:2] == ['install']:
    subprocess.check_call([sys.executable, 'build.py'],
                          cwd=os.path.abspath(os.path.dirname(__file__)))
else:
    print("Usage: setup.py install\n\nThis will just run build.py --- useful for readthdocs.org\n")
    sys.exit(1)
