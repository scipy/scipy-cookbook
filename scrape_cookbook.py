import glob
import os
import pweave
import shutil

os.chdir('pages')


dirs = glob.glob('CookBook(2f)*')

n = len(dirs)

for i in range(n):
    #try:    
    base = dirs[i]
        #base = "Cookbook(2f)KalmanFiltering"
    current = open(base +  "/current", "r").read().strip()
    source = "%s/revisions/%s" %  (base, current)
    attachments = "%s/attachments" % base
    new_attachments = "../originals/" + base.replace("Cookbook(2f)", "") + "_attachments"
    try:
        shutil.copytree(attachments, new_attachments)
    except:
        pass
