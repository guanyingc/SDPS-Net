from __future__ import division
import os
import numpy as np
import scipy.io as sio
#from scipy.ndimage import imread
from imageio import imread

import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util
np.random.seed(0)

class UPS_DiLiGenT_main(data.Dataset):
    def __init__(self, args, split='train'):
        self.root  = os.path.join(args.bm_dir)
        self.split = split
        self.args  = args
        self.objs  = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.names = util.readList(os.path.join(self.root, 'names.txt'),   sort=False)
        self.l_dir = util.light_source_directions()
        args.log.printWrite('[%s Data] \t%d objs %d lights. Root: %s' % 
                (split, len(self.objs), len(self.names), self.root))
        self.ints = {}
        ints_name = 'light_intensities.txt'
        print('Files for intensity: %s' % (ints_name))
        for obj in self.objs:
            self.ints[obj] = np.genfromtxt(os.path.join(self.root, obj, ints_name))

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def __getitem__(self, index):
        np.random.seed(index)
        obj = self.objs[index]
        select_idx = range(len(self.names))

        img_list = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]
        ints = [np.diag(1 / self.ints[obj][i]) for i in select_idx]
        dirs = self.l_dir[select_idx]

        normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            if self.args.in_light and not self.args.int_aug:
                img = np.dot(img, ints[idx])
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        mask = self._getMask(obj)
        if self.args.test_resc:
            img, normal = pms_transforms.rescale(img, normal, [self.args.test_h, self.args.test_w])
            mask = pms_transforms.rescaleSingle(mask, [self.args.test_h, self.args.test_w])

        img = img * mask.repeat(img.shape[2], 2)

        norm = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)

        item = {'normal': normal, 'img': img, 'mask': mask}

        downsample = 4 
        for k in item.keys():
            item[k] = pms_transforms.imgSizeToFactorOfK(item[k], downsample)

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        item['dirs'] = torch.from_numpy(dirs).view(-1, 1, 1).float()
        item['ints'] = torch.from_numpy(self.ints[obj][select_idx]).view(-1, 1, 1).float()

        item['obj'] = obj
        item['path'] = os.path.join(self.root, obj)
        return item

    def __len__(self):
        return len(self.objs)
