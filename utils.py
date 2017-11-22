from __future__ import print_function, division

import argparse
import os, sys
import os.path as osp
import parse
import numpy as np
from time import time
from PIL import Image
import contextlib
import yaml
import shutil
from collections import OrderedDict
import matplotlib.pyplot as plt


class NestedDict(OrderedDict):
    def __missing__(self, key):
        self[key] = NestedDict()
        return self[key]

    def is_leaf(self, key):
        if key not in self:
            raise ValueError('Key {} not found.'.format(key))
        return not isinstance(self[key], NestedDict)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isint(s):
    """test if string is an int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def mean(l):
    return sum(l) / len(l)


# https://stackoverflow.com/questions/2860153/how-do-i-get-the-parent-directory-in-python
def parent_dir(path):
    return os.path.split(os.path.normpath(path))


def get_saves(dest_dir, pattern='{:d}'):
    if not os.path.exists(dest_dir):
        return []
    files = os.listdir(dest_dir)
    p = parse.compile(pattern)
    saves = []
    for s in files:
        parsed = p.parse(s)
        if parsed:
            saves.append((parsed[0], os.path.join(dest_dir, s)))
    return sorted(saves)


def ifmakedirs(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def dict_str(data):
    return yaml.dump(data)#, default_flow_style=False)


def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
                (0.5, 0.5, 0.5), (0.25, 0, 0), (0.75, 0, 0), (0.25,
                                                              0.5, 0), (0.75, 0.5, 0), (0.25, 0, 0.5),
                (0.75, 0, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5,
                                                   0.5), (0, 0.25, 0), (0.5, 0.25, 0), (0, 0.75, 0),
                (0.5, 0.75, 0), (0, 0.25, 0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()


# https://stackoverflow.com/questions/2891790/how-to-pretty-printing-a-numpy-array-without-scientific-notation-and-with-given
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


@contextlib.contextmanager
def write(path, flag='w'):
    """Delayed write operation to avoid overwriting data in case of failure."""
    tmp_path = path + '.tmp'
    try:
        f = open(tmp_path, flag)
        yield f
        f.close()
        shutil.move(tmp_path, path)
    finally:
        f.close()



def discrete_matshow(data, cmap=None, cbar=True):
    # https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    if cmap is None:
        cmap = plt.get_cmap('RdBu', np.max(data)-np.min(data)+1)
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data)-.5,
                      vmax=np.max(data)+.5, fignum=False)
    if cbar:
        cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data)+1),
                       fraction=0.046, pad=0.04)


def Tictoc():
    start_stack = []
    start_named = {}

    def tic(name=None):
        if name is None:
            start_stack.append(time())
        else:
            start_named[name] = time()

    def toc(name=None):
        if name is None:
            start = start_stack.pop()
        else:
            start = start_named.pop(name)
        elapsed = time() - start
        return elapsed
    return tic, toc
