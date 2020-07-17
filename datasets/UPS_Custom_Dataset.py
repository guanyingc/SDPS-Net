from __future__ import division
import os
import numpy as np
import scipy.io as sio
from imageio import imread

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class UPS_Custom_Dataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.root   = os.path.join(args.bm_dir)
        self.split  = split
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        args.log.printWrite('[%s Data] \t%d objs. Root: %s' % (split, len(self.objs), self.root))

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def __getitem__(self, index):
        obj   = self.objs[index]
        names  = util.readList(os.path.join(self.root, obj, 'names.txt'))
        img_list   = [os.path.join(self.root, obj, names[i]) for i in range(len(names))]

        if self.args.have_l_dirs:
            dirs = np.genfromtxt(os.path.join(self.root, obj, 'light_directions.txt'))
        else:
            dirs = np.zeros((len(names), 3))
            dirs[:,2] = 1
        
        if self.args.have_l_ints:
            ints = np.genfromtxt(os.path.join(self.root, obj, 'light_intensities.txt'))
        else:
            ints = np.ones((len(names), 3))

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            imgs.append(img)
        img = np.concatenate(imgs, 2)
        h, w, c = img.shape

        if self.args.have_gt_n:
            normal_path = os.path.join(self.root, obj, 'normal.png')
            normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        else:
            normal = np.zeros((h, w, 3))

        mask = self._getMask(obj)
        img  = img * mask.repeat(img.shape[2], 2)

        item = {'normal': normal, 'img': img, 'mask': mask}

        downsample = 4 
        for k in item.keys():
            item[k] = pms_transforms.imgSizeToFactorOfK(item[k], downsample)

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(ints).view(-1, 1, 1).float()
        item['obj'] = obj
        item['path'] = os.path.join(self.root, obj)
        return item

    def __len__(self):
        return len(self.objs)
