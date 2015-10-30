#!/usr/bin/env python
"""
build.py

Build HTML output
"""
from __future__ import absolute_import, division, print_function

import re
import os
import argparse
import subprocess
import shutil


def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument('--html', action='store_true', help="Build HTML output")
    args = p.parse_args()

    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    dst_path = os.path.join('docs', 'items')
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    shutil.copytree(os.path.join('ipython', 'attachments'),
                    os.path.join(dst_path, 'attachments'))

    tags = parse_wiki_legacy_tags()
    titles, tags_new = generate_files(dst_path=dst_path)

    tags.update(tags_new)
    write_index(dst_path, titles, tags)

    if args.html:
        subprocess.check_call(['sphinx-build', '-b', 'html', '.', '_build/html'],
                              cwd='docs')


def write_index(dst_path, titles, tags):
    index_rst = os.path.join(dst_path, 'index.txt')

    # Write doctree
    toctree_items = []
    index_text = []

    # Fill in missing tags
    for fn in titles.keys():
        if fn not in tags or not tags[fn]:
            tags[fn] = ['Other examples']

    # Count tags
    tag_counts = {}
    for fn, tagset in tags.items():
        for tag in tagset:
            if tag not in tag_counts:
                tag_counts[tag] = 1
            else:
                tag_counts[tag] += 1
    tag_counts['Outdated'] = 1e99

    # Generate tree
    tag_sets = {}
    for fn, tagset in tags.items():
        tagset = list(set(tagset))
        tagset.sort(key=lambda tag: -tag_counts[tag])
        if 'Outdated' in tagset:
            tagset = ['Outdated']
        tag_id = " / ".join(tagset[:2])
        tag_sets.setdefault(tag_id, []).append(fn)

    tag_sets = list(tag_sets.items())
    def tag_set_sort(item):
        return (1 if 'Outdated' in item[0] else 0,
                item)
    tag_sets.sort(key=tag_set_sort)

    # Produce output
    for tag_id, fns in tag_sets:
        fns = [fn for fn in fns if fn in titles]
        if not fns:
            continue
        fns.sort(key=lambda fn: titles[fn])

        section_base_fn = re.sub('_+', '_', re.sub('[^a-z0-9_]', "_", "_" + tag_id.lower())).strip('_')
        section_fn = os.path.join(dst_path, section_base_fn + '.rst')
        toctree_items.append(section_base_fn)

        index_text.append("\n{0}\n{1}\n\n".format(tag_id, "-"*len(tag_id)))
        for fn in fns:
            index_text.append(":doc:`{0} <items/{1}>`\n".format(titles[fn], fn))

        with open(section_fn, 'w') as f:
            f.write("{0}\n{1}\n\n".format(tag_id, "="*len(tag_id)))
            f.write(".. toctree::\n"
                    "   :maxdepth: 1\n\n")
            for fn in fns:
                f.write("   {0}\n".format(fn))

    # Write index
    with open(index_rst, 'w') as f:
        f.write(".. toctree::\n"
                "   :maxdepth: 1\n"
                "   :hidden:\n\n")
        for fn in toctree_items:
            f.write("   items/%s\n" % (fn,))
        f.write("\n\n")
        f.write('.. raw:: html\n\n   <div id="cookbook-index">\n\n')
        f.write("".join(index_text))
        f.write('\n\n.. raw:: html\n\n   </div>\n')
        f.close()


def generate_files(dst_path):
    titles = {}
    tags = {}

    for fn in sorted(os.listdir('ipython')):
        if not fn.endswith('.ipynb'):
            continue
        fn = os.path.join('ipython', fn)
        title, tagset = parse_file(dst_path, fn)
        basename = os.path.splitext(os.path.basename(fn))[0]
        titles[basename] = title
        if tagset:
            tags[basename] = tagset

    return titles, tags


def parse_file(dst_path, fn):
    print(fn)
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'rst', os.path.abspath(fn)],
                          cwd=dst_path,
                          stderr=subprocess.PIPE)

    basename = os.path.splitext(os.path.basename(fn))[0]
    rst_fn = os.path.join(dst_path, basename + '.rst')

    title = None
    tags = set()

    with open(rst_fn, 'r') as f:
        prev_line = ''
        for line in f:
            line = line.strip()
            m = re.match('^===+\s*$', line)
            m2 = re.match('^---+\s*$', line)
            if m or m2:
                if prev_line and len(line) >= len(prev_line) and not title:
                    title = prev_line.strip()
                continue

            m = re.match('^TAGS:\s*(.*)\s*$', line)
            if m:
                tag_line = m.group(1).strip().replace(';', ',')
                tags.update(tag_line.split())
                continue

            prev_line = line

    with open(rst_fn, 'r') as f:
        text = f.read()
    if not title:
        text = "{0}\n{1}\n\n{2}".format(basename, "="*len(basename), text)
        title = basename
    text = re.sub(r'`(.*?) <files/(attachments/.*?)>`__',
                  r':download:`\1 <\2>`',
                  text,
                  flags=re.M)
    text = re.sub(r'^TAGS:.*$', '', text, flags=re.M)
    text = re.sub(r'(figure|image):: files/attachments/', r'\1:: attachments/', text, flags=re.M)
    text = re.sub(r' <files/attachments/', r' <attachments/', text, flags=re.M)
    text = re.sub(r'.. parsed-literal::', r'.. parsed-literal::\n   :class: ipy-out', text, flags=re.M)
    text = re.sub(r'`([^`<]*)\s+<(?!attachments/)([^:.>]*?)(?:.html)?>`__', r':doc:`\1 <items/\2>`', text, flags=re.M)
    with open(rst_fn, 'w') as f:
        f.write(text)
    del text

    attach_dir = os.path.join('ipython', 'attachments', basename)
    if os.path.isdir(attach_dir) and len(os.listdir(attach_dir)) > 0:
        with open(rst_fn, 'a') as f:
            f.write("""

Attachments
-----------
""")
            images = []
            for fn in sorted(os.listdir(attach_dir)):
                if os.path.isfile(os.path.join(attach_dir, fn)):
                    if os.path.splitext(fn.lower())[1] in ('.png', '.jpg', '.jpeg'):
                        images.append(fn)
                    f.write('- :download:`%s <attachments/%s/%s>`\n' % (
                        fn, basename, fn))

            f.write("\n\n")
            for fn in images:
                f.write('.. image:: attachments/%s/%s\n' % (
                    basename, fn))

    return title, tags


def parse_wiki_legacy_tags():
    tags = [None, None]
    items = {}

    with open('wiki-legacy-tags.txt', 'r') as f:
        prev_line = None

        for line in f:
            if re.match('^====+\s*$', line):
                tags[0] = prev_line.strip()
                tags[1] = None
                continue

            if re.match('^----+\s*$', line):
                tags[1] = prev_line.strip()
                continue

            prev_line = line

            m = re.search(r'\[\[(.*?)(?:\|.*)?\]\]', line)
            if m:
                name = m.group(1).strip()
                name = re.sub('Cookbook/', '', name)
                name = re.sub('^/', '', name)
                name = name.replace('/', '_').replace(' ', '_')

                fn = os.path.join('ipython', name + '.ipynb')
                if os.path.isfile(fn):
                    basename = os.path.splitext(os.path.basename(fn))[0]
                    items.setdefault(basename, set()).update([x for x in tags if x])
                continue

    return items


if __name__ == "__main__":
    main()
