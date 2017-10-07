"""
generate-legacy-dates.py WIKI_PAGES_PATH

"""

import os
import re
import sys

for fn in os.listdir('ipython'):
    base, ext = os.path.splitext(fn)
    if ext != '.ipynb':
        continue

    pth_base = base
    pth_base = re.sub('^Matplotlib_', 'Matplotlib(2f)', pth_base)
    pth_base = pth_base.replace('MayaVi_', 'MayaVi(2f)')
    pth_base = pth_base.replace('KDTree_example', 'KDTree')
    pth_base = pth_base.replace('PIL_example', 'PIL')
    pth_base = pth_base.replace('ScriptingMayavi2_', 'ScriptingMayavi2(2f)')
    pth_base = pth_base.replace('C_Extensions_NumPy_arrays', 'C_Extensions(2f)NumPy_arrays')
    pth_base = pth_base.replace('Theoretical_Ecology_', 'Theoretical_Ecology(2f)')
    pth_base = pth_base.replace('FortranIO_', 'FortranIO(2f)')
    pth_base = pth_base.replace('TimeSeries_', 'TimeSeries(2f)')

    editlog = os.path.abspath(sys.argv[1]) + '/'
    if base not in ('ParallelProgramming', 'PerformancePython'):
        editlog += 'Cookbook(2f)'
    editlog += pth_base + '/edit-log'

    if not os.path.isfile(editlog):
        continue

    with open(editlog, 'r') as f:
        lines = f.readlines()

    timestamps = []
    for line in lines:
        line = line.strip().split()
        if not line:
            continue
        timestamps.append(int(float(line[0]) / 1e6))

    print(base, min(timestamps), max(timestamps))
