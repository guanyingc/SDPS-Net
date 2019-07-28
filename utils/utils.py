import os 
import numpy as np
from imageio import imread, imsave

def makeFile(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def makeFiles(f_list):
    for f in f_list:
        makeFile(f)

def emptyFile(name):
    with open(name, 'w') as f:
        f.write(' ')

def dictToString(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def checkIfInList(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def readList(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists
