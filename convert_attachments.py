import os
import glob

ext = ".Pnw"
target = 'attachments/'

os.chdir('converted')
docs = glob.glob('*' + ext)


def convert_attachments(lines, path):
    n = len(lines)
    new = []
    for i in range(n):
        line = lines[i]
        if "attachment:" in line:
            r = line.replace("attachment:", ".. image:: " + path)
            r = r.replace("\\_", "_")
        elif "inline:" in line:
            r = line.replace(" ", "\n\n")
            r = r.replace("inline:", ".. image:: " + path)
            r = r.replace("\\_", "_")
        else:
            r = line
        new.append(r)    
    return("".join(new))


for doc in docs[:]:
    lines = open(doc).readlines()
    impath = doc.replace(ext, "") + "_attachments/"
    converted = convert_attachments(lines, impath)
    new = open('../' + target  + doc, "w")
    new.write(converted)
    new.close()