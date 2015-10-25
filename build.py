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
import json
import lxml.html
import shutil


def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument('--reparse', action='store_true')
    args = p.parse_args()

    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    write_tags(reparse=args.reparse)
    do_gh_pages()


def do_gh_pages():
    if os.path.exists('build'):
        shutil.rmtree('build')
        os.makedirs('build')

    subprocess.check_call(['git', 'clone', '-b', 'gh-pages', '.', 'build'])
    subprocess.check_call(['git', '-C', 'build', 'rm', '-rf', '.'])

    for fn in os.listdir('www'):
        if fn.endswith('~'):
            continue
        shutil.copyfile(os.path.join('www', fn),
                        os.path.join('build', fn))
        
    subprocess.check_call(['git', '-C', 'build', 'add', '.'])
    subprocess.check_call(['git', '-C', 'build', 'commit', '--amend', '-m', 'Rebuild web site'])
    subprocess.check_call(['git', 'fetch', 'build'])
    subprocess.check_call(['git', 'branch', '-f', 'gh-pages', 'FETCH_HEAD'])

    print("gh-pages branch ready for git push -f")


def write_tags(reparse=False):
    data_json = os.path.join('www', 'data.json')

    if reparse or not os.path.isfile(data_json):
        tags = parse_wiki_legacy_tags()
        titles = parse_files()
        data = {
            'tags': tags,
            'titles': titles
        }
        with open(data_json, 'w') as f:
            json.dump(data, f)
    else:
        with open(data_json, 'r') as f:
            data = json.load(f)
        titles = data['titles']
        tags = data['tags']


def parse_files():
    titles = {}

    for fn in sorted(os.listdir('ipython')):
        if not fn.endswith('.ipynb'):
            continue
        fn = os.path.join('ipython', fn)
        title, html = parse_file(fn)
        titles[os.path.basename(fn)] = title

    return titles


def parse_file(fn):
    print(fn)
    html = subprocess.check_output(['ipython', 'nbconvert', '--to', 'html',
                                    '--stdout', fn], stderr=subprocess.PIPE)
    tree = lxml.html.fromstring(html)

    h1 = tree.xpath('//h1')
    if h1:
        title = h1[0].text.strip()
    else:
        title = os.path.splitext(os.path.basename(fn))[0].replace('_', ' ').strip()

    return title, html


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
                    items[os.path.basename(fn)] = list(set(x for x in tags if x))
                continue

    return items


if __name__ == "__main__":
    main()
