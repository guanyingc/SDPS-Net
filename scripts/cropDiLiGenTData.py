import os, argparse, sys, shutil, glob
import numpy as np
from imageio import imread, imsave
import scipy.io as sio

root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(root_path)
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',  default='data/datasets/DiLiGenT/pmsData')
parser.add_argument('--obj_list',   default='objects.txt')
parser.add_argument('--suffix',     default='crop')
parser.add_argument('--file_ext',   default='.png')
parser.add_argument('--normal_name',default='Normal_gt.png')
parser.add_argument('--mask_name',  default='mask.png')
parser.add_argument('--n_key',      default='Normal_gt')
parser.add_argument('--pad',        default=15, type=int)
args = parser.parse_args()

def getSaveDir():
    dirName  = os.path.dirname(args.input_dir)
    save_dir = '%s_%s' % (args.input_dir, args.suffix) 
    utils.makeFile(save_dir)
    print('Output dir: %s\n' % save_dir)
    return save_dir

def getBBoxCompact(mask):
    index = np.where(mask != 0)
    t, b, l , r = index[0].min(), index[0].max(), index[1].min(), index[1].max()
    h, w = b - t + 2 * args.pad, r - l + 2 * args.pad
    t = max(0, t - args.pad)
    b = t + h 
    l = max(0, l - args.pad)
    r = l + w 
    if h % 4 != 0: 
        pad = 4 - h % 4
        b += pad; h += pad
    if w % 4 != 0: 
        pad = 4 - w % 4
        r += pad; w += pad
    return l, r, t, b, h, w

def loadMaskNormal(d):
    mask   = imread(os.path.join(args.input_dir, d, args.mask_name))
    try:
        normal = imread(os.path.join(args.input_dir, d, args.normal_name))
    except IOError:
        normal = imread(os.path.join(args.input_dir, d, 'normal_gt.png'))
    n_mat  = sio.loadmat(os.path.join(args.input_dir, d, 'Normal_gt.mat'))[args.n_key]
    h, w, c = normal.shape
    print('Processing Objects: %s' % d, mask.shape)
    if mask.ndim < 3:
        mask = mask.reshape(h, w, 1).repeat(3, 2)
    return mask, normal, n_mat

def copyTXT(d):
    txt = glob.glob(os.path.join(args.input_dir, '*.txt'))
    for t in txt:
        name = os.path.basename(t)
        shutil.copy(t, os.path.join(args.save_dir, name))

    txt = glob.glob(os.path.join(args.input_dir, d, '*.txt'))
    for t in txt:
        name = os.path.basename(t)
        shutil.copy(t, os.path.join(args.save_dir, d, name))

if __name__ == '__main__':
    print('Input dir: %s\n' % args.input_dir)
    args.save_dir = getSaveDir()

    dir_list  = utils.readList(os.path.join(args.input_dir, args.obj_list))
    name_list = utils.readList(os.path.join(args.input_dir, 'names.txt'))
    max_h, max_w = 0, 0
    crop_list = open(os.path.join(args.save_dir, 'crop.txt'), 'w')
    for d in dir_list:
        utils.makeFile(os.path.join(args.save_dir, d))
        mask, normal, n_mat = loadMaskNormal(d)
        l, r, t, b, h, w = getBBoxCompact(mask[:,:,0] / 255)
        crop_list.write('%d %d %d %d %d %d\n' % (mask.shape[0], mask.shape[1], l, r, t, b))

        max_h = h if h > max_h else max_h
        max_w = w if w > max_w else max_w
        print('\t BBox L %d R %d T %d B %d, H:%d W:%d, Padded: %d %d' % 
                (l, r, t, b, h, w, r - l, b - t)) 
        imsave(os.path.join(args.save_dir, d, args.mask_name), mask[t:b, l:r, :])
        imsave(os.path.join(args.save_dir, d, args.normal_name), normal[t:b, l:r, :])
        sio.savemat(os.path.join(args.save_dir, d, 'Normal_gt.mat'), 
                {args.n_key: n_mat[t:b, l:r, :]} ,do_compression=True)
        copyTXT(d)
        intens = np.genfromtxt(os.path.join(args.input_dir, d, 'light_intensities.txt'))
        for idx, name in enumerate(name_list):
            #print('Process img %d/%d' % (idx+1, len(name_list)))
            img = imread(os.path.join(args.input_dir, d, name))
            img = img[t:b, l:r, :]
            imsave(os.path.join(args.save_dir, d, name), img.astype(np.uint8))
    print('Max H %d, Max %d' % (max_h, max_w))
    crop_list.close()
