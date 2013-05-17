import os
import glob
import pweave
import shutil

os.chdir('originals')
docs = glob.glob('*.txt')

for doc in docs:    
    pweave.convert(doc, "wiki", "noweb", "-f mediawiki -t rst")
    new = "../converted/" + doc.replace(".txt", ".Pnw")
    shutil.move(doc.replace(".txt", '.Pnw'), new)
    pweave.convert(doc, "wiki", "ipython", "-f mediawiki -t markdown")
    new = "../tmp/" + doc.replace(".txt", ".py")
    shutil.move(doc.replace(".txt", '.py'), new)