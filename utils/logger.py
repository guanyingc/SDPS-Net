import datetime, time, os
import numpy as np
import torch
import torchvision.utils as vutils
import scipy.io as sio
from . import utils

import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
plt.rcParams["figure.figsize"] = (5,8)

class Logger(object):
    def __init__(self, args):
        self.times = {'init': time.time()}
        if args.make_dir:
            self._setupDirs(args)
        self.args = args
        args.log  = self
        self.printArgs()

    def printArgs(self):
        strs = '------------ Options -------------\n'
        strs += '{}'.format(utils.dictToString(vars(self.args)))
        strs += '-------------- End ----------------\n'
        self.printWrite(strs)

    def _addArguments(self, args):
        info = ''
        if hasattr(args, 'run_model') and args.run_model:
            info += '_run_model,%s' % os.path.basename(args.retrain).split('.')[0]
        arg_var  = vars(args)
        for k in args.str_keys:  
            info = '{0},{1}'.format(info, arg_var[k])
        for k in args.val_keys:  
            var_key = k[:2] + '_' + k[-1]
            info = '{0},{1}-{2}'.format(info, var_key, arg_var[k])
        for k in args.bool_keys: 
            info = '{0},{1}'.format(info, k) if arg_var[k] else info 
        return info

    def _setupDirs(self, args):
        date_now = datetime.datetime.now()
        dir_name = '%d-%d' % (date_now.month, date_now.day)
        dir_name += (',%s' % args.suffix) if args.suffix else ''
        dir_name += self._addArguments(args) 
        dir_name += ',DEBUG' if args.debug else ''

        self._checkPath(args, dir_name)
        file_dir = os.path.join(args.log_dir, '%s,%s' % (dir_name, date_now.strftime('%H:%M:%S')))
        self.log_fie = open(file_dir, 'w')
        return 

    def _checkPath(self, args, dir_name):
        if hasattr(args, 'run_model') and args.run_model:
            log_root = os.path.join(os.path.dirname(args.retrain), dir_name)
            args.log_dir = log_root
            sub_dirs = ['test']
        else:
            if args.resume and os.path.isfile(args.resume):
                log_root = os.path.join(os.path.dirname(os.path.dirname(args.resume)), dir_name)
            else:
                if args.debug:
                    dir_name = 'DEBUG/' + dir_name
                log_root = os.path.join(args.save_root, args.dataset, args.item, dir_name)
            args.log_dir = os.path.join(log_root, 'logdir')
            args.cp_dir  = os.path.join(log_root, 'checkpointdir')
            utils.makeFiles([args.log_dir, args.cp_dir])
            sub_dirs = ['train', 'val'] 
        for sub_dir in sub_dirs:
            utils.makeFiles([os.path.join(args.log_dir, sub_dir, 'Images')])

    def printWrite(self, strs):
        print('%s' % strs)
        if self.args.make_dir:
            self.log_fie.write('%s\n' % strs)
            self.log_fie.flush()

    def getTimeInfo(self, epoch, iters, batch):
        time_elapsed = (time.time() - self.times['init']) / 3600.0
        total_iters  = (self.args.epochs - self.args.start_epoch + 1) * batch
        cur_iters    = (epoch - self.args.start_epoch) * batch + iters
        time_total   = time_elapsed * (float(total_iters) / cur_iters)
        return time_elapsed, time_total

    def printItersSummary(self, opt):
        epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
        strs = ' | {}'.format(str.upper(opt['split']))
        strs += ' Iter [{}/{}] Epoch [{}/{}]'.format(iters, batch, epoch, self.args.epochs)
        if opt['split'] == 'train': 
            time_elapsed, time_total = self.getTimeInfo(epoch, iters, batch) # Buggy for test
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed, time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
        self.printWrite(strs)
        if 'timer' in opt.keys(): 
            self.printWrite(opt['timer'].timeToString())
        if 'recorder' in opt.keys(): 
            self.printWrite(opt['recorder'].iterRecToString(opt['split'], epoch))

    def printEpochSummary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        self.printWrite('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        self.printWrite(opt['recorder'].epochRecToString(split, epoch))

    def convertToSameSize(self, t_list):
        shape = (t_list[0].shape[0], 3, t_list[0].shape[2], t_list[0].shape[3])
        for i, tensor in enumerate(t_list):
            n, c, h, w = tensor.shape
            if tensor.shape[1] != shape[1]: # check channel
                t_list[i] = tensor.expand((n, shape[1], h, w))
            if h == shape[2] and w == shape[3]:
                continue
            t_list[i] = torch.nn.functional.upsample(tensor, [shape[2], shape[3]], mode='bilinear')
        return t_list

    def getSaveDir(self, split, epoch):
        save_dir = os.path.join(self.args.log_dir, split, 'Images')
        run_model = hasattr(self.args, 'run_model') and self.args.run_model
        if not run_model and epoch > 0:
            save_dir = os.path.join(save_dir, str(epoch))
        utils.makeFile(save_dir)
        return save_dir

    def splitMulitChannel(self, t_list, max_save_n = 8):
        new_list = []
        for tensor in t_list:
            if tensor.shape[1] > 3:
                num = 3
                new_list += torch.split(tensor, num, 1)[:max_save_n]
            else:
                new_list.append(tensor)
        return new_list

    def saveSplit(self, res, save_prefix):
        n, c, h, w = res.shape
        for i in range(n):
            vutils.save_image(res[i], save_prefix + '_%d.png' % (i))

    def saveImgResults(self, results, split, epoch, iters, nrow, error=''):
        max_save_n = self.args.test_save_n if split == 'test' else self.args.train_save_n
        res = [img.cpu() for img in results]
        res = self.splitMulitChannel(res, max_save_n)
        res = torch.cat(self.convertToSameSize(res))
        save_dir = self.getSaveDir(split, epoch)
        save_prefix = os.path.join(save_dir, '%d_%d' % (epoch, iters))
        save_prefix += ('_%s' % error) if error != '' else ''
        if self.args.save_split: 
            self.saveSplit(res, save_prefix)
        else:
            vutils.save_image(res, save_prefix + '_out.png', nrow=nrow)

    def plotCurves(self, recorder, split='train', epoch=-1, intv=1):
        dict_of_array = recorder.recordToDictOfArray(split, epoch, intv)
        save_dir = os.path.join(self.args.log_dir, split)
        if epoch < 0:
            save_dir = self.args.log_dir
            save_name = '%s_Summary.png' % (split)
        else:
            save_name = '%s_epoch_%d.png' % (split, epoch)

        classes = ['loss', 'acc', 'err', 'lr', 'ratio']
        classes = utils.checkIfInList(classes, dict_of_array.keys())
        if len(classes) == 0: return

        for idx, c in enumerate(classes):
            plt.subplot(len(classes), 1, idx+1)
            plt.grid()
            legends = []
            for k in dict_of_array.keys():
                if (c in k.lower()) and not k.endswith('_x'):
                    plt.plot(dict_of_array[k+'_x'], dict_of_array[k])
                    legends.append(k)
            if len(legends) != 0:
                plt.legend(legends, bbox_to_anchor=(0.5, 1.05), loc='upper center', 
                            ncol=len(legends), prop=fontP)
                plt.title(c)
                if epoch < 0: plt.xlabel('Epoch') 
                else: plt.xlabel('Iters')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, save_name))
        plt.clf()
