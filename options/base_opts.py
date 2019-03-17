import argparse
import os
import torch

class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Trainining Dataset ####
        self.parser.add_argument('--dataset',     default='UPS_Synth_Dataset')
        self.parser.add_argument('--data_dir',    default='data/datasets/PS_Blobby_Dataset')
        self.parser.add_argument('--data_dir2',   default='data/datasets/PS_Sculpture_Dataset')
        self.parser.add_argument('--concat_data', default=True, action='store_false')
        self.parser.add_argument('--l_suffix',    default='_mtrl.txt')

        #### Training Data and Preprocessing Arguments ####
        self.parser.add_argument('--rescale',     default=True,  action='store_false')
        self.parser.add_argument('--rand_sc',     default=True,  action='store_false')
        self.parser.add_argument('--scale_h',     default=128,   type=int)
        self.parser.add_argument('--scale_w',     default=128,   type=int)
        self.parser.add_argument('--crop',        default=True,  action='store_false')
        self.parser.add_argument('--crop_h',      default=128,   type=int)
        self.parser.add_argument('--crop_w',      default=128,   type=int)
        self.parser.add_argument('--test_h',      default=128,   type=int)
        self.parser.add_argument('--test_w',      default=128,   type=int)
        self.parser.add_argument('--test_resc',   default=True,  action='store_false')
        self.parser.add_argument('--int_aug',     default=True,  action='store_false')
        self.parser.add_argument('--noise_aug',   default=True,  action='store_false')
        self.parser.add_argument('--noise',       default=0.05,  type=float)
        self.parser.add_argument('--color_aug',   default=True,  action='store_false')
        self.parser.add_argument('--color_ratio', default=3,     type=float)
        self.parser.add_argument('--normalize',   default=False, action='store_true')

        #### Device Arguments ####
        self.parser.add_argument('--cuda',        default=True,  action='store_false')
        self.parser.add_argument('--multi_gpu',   default=False, action='store_true')
        self.parser.add_argument('--time_sync',   default=False, action='store_true')
        self.parser.add_argument('--workers',     default=8,     type=int)
        self.parser.add_argument('--seed',        default=0,     type=int)

        #### Stage 1 Model Arguments ####
        self.parser.add_argument('--dirs_cls',    default=36,    type=int)
        self.parser.add_argument('--ints_cls',    default=20,    type=int)
        self.parser.add_argument('--dir_int',     default=False, action='store_true')
        self.parser.add_argument('--model',       default='LCNet')
        self.parser.add_argument('--fuse_type',   default='max')
        self.parser.add_argument('--in_img_num',  default=32,    type=int)
        self.parser.add_argument('--s1_est_n',    default=False, action='store_true')
        self.parser.add_argument('--s1_est_d',    default=True,  action='store_false')
        self.parser.add_argument('--s1_est_i',    default=True,  action='store_false')
        self.parser.add_argument('--in_light',    default=False, action='store_true')
        self.parser.add_argument('--in_mask',     default=True,  action='store_false')
        self.parser.add_argument('--use_BN',      default=False, action='store_true')
        self.parser.add_argument('--resume',      default=None)
        self.parser.add_argument('--retrain',     default=None)
        self.parser.add_argument('--save_intv',   default=1,     type=int)

        #### Stage 2 Model Arguments ####
        self.parser.add_argument('--stage2',      default=False, action='store_true')
        self.parser.add_argument('--model_s2',    default='NENet')
        self.parser.add_argument('--retrain_s2',  default=None)
        self.parser.add_argument('--s2_est_n',    default=True,  action='store_false')
        self.parser.add_argument('--s2_est_i',    default=False, action='store_true')
        self.parser.add_argument('--s2_est_d',    default=False, action='store_true')
        self.parser.add_argument('--s2_in_light', default=True,  action='store_false')

        #### Displaying Arguments ####
        self.parser.add_argument('--train_disp',    default=20,  type=int)
        self.parser.add_argument('--train_save',    default=200, type=int)
        self.parser.add_argument('--val_intv',      default=1,   type=int)
        self.parser.add_argument('--val_disp',      default=1,   type=int)
        self.parser.add_argument('--val_save',      default=1,   type=int)
        self.parser.add_argument('--max_train_iter',default=-1,  type=int)
        self.parser.add_argument('--max_val_iter',  default=-1,  type=int)
        self.parser.add_argument('--max_test_iter', default=-1,  type=int)
        self.parser.add_argument('--train_save_n',  default=4,   type=int)
        self.parser.add_argument('--test_save_n',   default=4,   type=int)

        #### Log Arguments ####
        self.parser.add_argument('--save_root',  default='data/logdir/')
        self.parser.add_argument('--item',       default='CVPR2019')
        self.parser.add_argument('--suffix',     default=None)
        self.parser.add_argument('--debug',      default=False, action='store_true')
        self.parser.add_argument('--make_dir',   default=True,  action='store_false')
        self.parser.add_argument('--save_split', default=False, action='store_true')

    def setDefault(self):
        if self.args.debug:
            self.args.train_disp = 1
            self.args.train_save = 1
            self.args.max_train_iter = 4 
            self.args.max_val_iter = 4
            self.args.max_test_iter = 4
            self.args.test_intv = 1
    def collectInfo(self):
        self.args.str_keys  = [
                'model', 'fuse_type', 'solver'
                ]
        self.args.val_keys  = [
                'batch', 'scale_h', 'crop_h', 'init_lr', 'normal_w', 
                'dir_w', 'ints_w', 'in_img_num', 'dirs_cls', 'ints_cls'
                ]
        self.args.bool_keys = [
                'use_BN', 'in_light', 'in_mask', 's1_est_n', 's1_est_d', 's1_est_i', 
                'color_aug', 'int_aug', 'concat_data', 'retrain', 'resume', 'stage2', 
                ] 

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
