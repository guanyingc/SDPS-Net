import time
import torch
from collections import OrderedDict

class Timer(object):
    def __init__(self, cuda_sync=False):
        self.timer = OrderedDict()
        self.cuda_sync = cuda_sync
        self.startTimer()

    def startTimer(self):
        self.iter_start = time.time()
        self.disp_start = time.time()

    def resetTimer(self):
        self.iter_start = time.time()
        self.disp_start = time.time()
        for key in self.timer.keys(): self.timer[key].reset()

    def updateTime(self, key):
        if key not in self.timer.keys(): self.timer[key] = AverageMeter()
        if self.cuda_sync: torch.cuda.synchronize()
        self.timer[key].update(time.time() - self.iter_start)
        self.iter_start = time.time()

    def timeToString(self, reset=True):
        strs = '\t [Time %.3fs] ' % (time.time() - self.disp_start)
        for key in self.timer.keys():
            if self.timer[key].sum < 1e-4: continue
            strs += '%s: %.3fs| ' % (key, self.timer[key].sum)
        self.resetTimer()
        return strs

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '%.3f' % (self.avg)
