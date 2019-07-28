import torch
import os
from utils import eval_utils

class Stage1ClsCrit(object): # First Stage, Light classification criterion
    def __init__(self, args):
        print('==> Using Stage1ClsCrit for lighting classification')
        self.s1_est_d = args.s1_est_d
        self.s1_est_i = args.s1_est_i
        self.ints_cls, self.dirs_cls = args.ints_cls, args.dirs_cls
        self.setupLightCrit(args)

    def setupLightCrit(self, args):
        args.log.printWrite('=> Using light criterion')
        if self.s1_est_d:
            self.dir_w = args.dir_w
            self.dirs_x_crit = torch.nn.CrossEntropyLoss()
            self.dirs_y_crit = torch.nn.CrossEntropyLoss()
            if args.cuda: 
                self.dirs_x_crit = self.dirs_x_crit.cuda()
                self.dirs_y_crit = self.dirs_y_crit.cuda()
        if self.s1_est_i:
            self.ints_w = args.ints_w
            self.ints_crit = torch.nn.CrossEntropyLoss()
            if args.cuda: self.ints_crit = self.ints_crit.cuda()

    def forward(self, output, target):
        self.loss = 0
        out_loss = {}
        if self.s1_est_d:
            est_dir_x, est_dir_y = output['dirs_x'], output['dirs_y']
            gt_dir_x, gt_dir_y = eval_utils.SphericalDirsToClass(target['dirs'], self.dirs_cls)
        
            dirs_x_loss = self.dirs_x_crit(est_dir_x, gt_dir_x)
            dirs_y_loss = self.dirs_y_crit(est_dir_y, gt_dir_y)

            out_loss['D_x_loss'] = dirs_x_loss.item()
            out_loss['D_y_loss'] = dirs_y_loss.item()
            self.loss += self.dir_w * (dirs_x_loss + dirs_y_loss)

        if self.s1_est_i:
            est_intens = output['ints']
            gt_ints = target['ints'].squeeze()[:, 0: target['ints'].shape[1]:3]
            gt_ints = torch.cat(torch.split(gt_ints, 1, 1), 0)
            gt_intens = eval_utils.LightIntsToClass(gt_ints, self.ints_cls)
            ints_loss = self.ints_crit(est_intens, gt_intens)
            out_loss['I_loss'] = ints_loss.item()
            self.loss += self.ints_w * ints_loss
        return out_loss
     
    def backward(self):
        self.loss.backward()

class Stage2Crit(object): # Second stage
    def __init__(self, args):
        self.s2_est_n = args.s2_est_n 
        self.s2_est_d = args.s2_est_d
        self.s2_est_i = args.s2_est_i
        self.setupLightCrit(args)
        if self.s2_est_n:
            self.setupNormalCrit(args)

    def setupLightCrit(self, args):
        args.log.printWrite('=> Using light criterion')
        if self.s2_est_d:
            self.dir_w = args.dir_w
            self.dirs_crit = torch.nn.CosineEmbeddingLoss()
            if args.cuda: self.dirs_crit = self.dirs_crit.cuda()
        if self.s2_est_i:
            self.ints_w = args.ints_w
            self.ints_crit = torch.nn.MSELoss()
            if args.cuda: self.ints_crit = self.ints_crit.cuda()

    def setupNormalCrit(self, args):
        args.log.printWrite('=> Using {} for criterion normal'.format(args.normal_loss))
        self.normal_w = args.normal_w
        if args.normal_loss == 'mse':
            self.n_crit = torch.nn.MSELoss()
        elif args.normal_loss == 'cos':
            self.n_crit = torch.nn.CosineEmbeddingLoss()
        else:
            raise Exception("=> Unknown Criterion '{}'".format(args.normal_loss))
        if args.cuda:
            self.n_crit = self.n_crit.cuda()

    def forward(self, output, target):
        self.loss = 0
        out_loss = {}

        if self.s2_est_d:
            d_est, d_tar = output['dirs'], target['dirs']
            d_num = d_tar.nelement() // d_tar.shape[1]
            if not hasattr(self, 'l_flag') or d_num != self.l_flag.nelement():
                self.l_flag = d_tar.data.new().resize_(d_num).fill_(1)
            dirs_loss = self.dirs_crit(d_est.squeeze(), d_tar, self.l_flag)
            self.loss += self.dir_w * dirs_loss
            out_loss['D_loss'] = dirs_loss.item()

        if self.s2_est_i:
            i_est, i_tar = output['ints'], target['ints']
            ints_loss  = self.ints_crit(i_est, i_tar.squeeze())
            self.loss += self.ints_w * ints_loss
            out_loss['I_loss'] = ints_loss.item()

        if self.s2_est_n:
            n_est, n_tar = output['n'], target['n']
            n_num = n_tar.nelement() // n_tar.shape[1]
            if not hasattr(self, 'n_flag') or n_num != self.n_flag.nelement():
                self.n_flag = n_tar.data.new().resize_(n_num).fill_(1)
            self.out_reshape = n_est.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.gt_reshape  = n_tar.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            normal_loss      = self.n_crit(self.out_reshape, self.gt_reshape, self.n_flag)
            self.loss += self.normal_w * normal_loss 
            out_loss['N_loss'] = normal_loss.item()
        return out_loss

    def backward(self):
        self.loss.backward()

def getOptimizer(args, params):
    args.log.printWrite('=> Using %s solver for optimization' % (args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    else:
        raise Exception("=> Unknown Optimizer %s" % (args.solver))
    return optimizer

def getLrScheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=args.milestones, gamma=args.lr_decay, last_epoch=args.start_epoch-2)
    return scheduler

def loadRecords(path, model, optimizer):
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' + path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] + 1
        records = records['records']
        print("=> loaded Records")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(path))
    return records, start_epoch

def configOptimizer(args, model):
    records = None
    optimizer = getOptimizer(args, model.parameters())
    if args.resume:
        args.log.printWrite("=> Resume loading checkpoint '{}'".format(args.resume))
        records, start_epoch = loadRecords(args.resume, model, optimizer)
        args.start_epoch = start_epoch
    scheduler = getLrScheduler(args, optimizer)
    return optimizer, scheduler, records
