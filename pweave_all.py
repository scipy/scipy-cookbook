
import os
import glob
import shutil

os.chdir('attachments')

files = glob.glob('*.Pnw')
n = 0


for doc in files:
    call = "Pweave -f sphinx %s" % doc
    os.system(call)
    rst_file = doc.replace('.Pnw', '.rst')
    shutil.move(rst_file, '../rst/' + rst_file) 
    n +=1 
    print n

    