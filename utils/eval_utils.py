import torch
import math
import numpy as np
from matplotlib import cm

def colorMap(diff):
    thres = 90
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = torch.from_numpy(cm.jet(diff_norm.numpy()))[:,:,:, :3]
    return diff_cm.permute(0,3,1,2).clone().float()

def calDirsAcc(gt_l, pred_l, data_batch=1):
    n, c = gt_l.shape
    pred_l = pred_l.view(n, c)
    dot_product = (gt_l * pred_l).sum(1).clamp(-1, 1)
    
    angular_err = torch.acos(dot_product) * 180.0 / math.pi
    l_err_mean  = angular_err.mean()
    return {'l_err_mean': l_err_mean.item()}, angular_err.squeeze()

def calIntsAcc(gt_i, pred_i, data_batch=1):
    n, c, h, w = gt_i.shape
    pred_i  = pred_i.view(n, c, h, w)
    ref_int = gt_i[:, :3].repeat(1, gt_i.shape[1] // 3, 1, 1)
    gt_i  = gt_i / ref_int
    scale = torch.gels(gt_i.view(-1, 1), pred_i.view(-1, 1))
    ints_ratio = (gt_i - scale[0][0] * pred_i).abs() / (gt_i + 1e-8)
    ints_error = torch.stack(ints_ratio.split(3, 1), 1).mean(2)
    return {'ints_ratio': ints_ratio.mean().item()}, ints_error.squeeze()
    
def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid  = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean = ang_valid.sum() / valid
    n_err_med  = ang_valid.median()
    n_acc_11   = (ang_valid < 11.25).sum().float() / valid
    n_acc_30   = (ang_valid < 30).sum().float() / valid
    n_acc_45   = (ang_valid < 45).sum().float() / valid

    angular_map = colorMap(angular_map.cpu().squeeze(1))
    value = {'n_err_mean': n_err_mean.item(), 
            'n_acc_11': n_acc_11.item(), 'n_acc_30': n_acc_30.item(), 'n_acc_45': n_acc_45.item()}
    angular_error_map = {'angular_map': angular_map}
    return value, angular_error_map

def SphericalDirsToClass(dirs, cls_num):
    theta = torch.atan(dirs[:,0] / (dirs[:,2] + 1e-8)) 
    denom = torch.sqrt(dirs[:,0] * dirs[:,0] + dirs[:,2] * dirs[:,2])
    phi = torch.atan(dirs[:,1] / (denom + 1e-8))
    theta = theta / np.pi * 180
    phi   = phi / np.pi * 180
    azimuth = ((theta + 90.0) / 180 * cls_num).clamp(0, cls_num-1).long()
    elevate = ((phi   + 90.0) / 180 * cls_num).clamp(0, cls_num-1).long()
    return azimuth, elevate

def SphericalClassToDirs(x_cls, y_cls, cls_num):
    theta = (x_cls.float() + 0.5) / cls_num * 180 - 90
    phi   = (y_cls.float() + 0.5) / cls_num * 180 - 90
    neg_x = theta < 0
    neg_y = phi < 0
    theta = theta.clamp(-90, 90) / 180.0 * np.pi
    phi   = phi.clamp(-90, 90) / 180.0 * np.pi

    tan2_phi   = pow(torch.tan(phi), 2)
    tan2_theta = pow(torch.tan(theta), 2)
    y = torch.sqrt(tan2_phi / (1 + tan2_phi))
    y[neg_y] = y[neg_y] * -1
    #y = torch.sin(phi)
    z = torch.sqrt((1 - y * y) / (1 + tan2_theta))
    x = z * torch.tan(theta)
    dirs = torch.stack([x,y,z], 1)
    dirs = dirs / dirs.norm(p=2, dim=1, keepdim=True)
    return dirs
    
def LightIntsToClass(ints, cls_num):
    ints = (ints - 0.2) / 1.8
    ints = (ints * cls_num).clamp(0, cls_num-1).long()
    return ints.view(-1)

def ClassToLightInts(cls, cls_num):
    ints = (cls.float() + 0.5) / cls_num * 1.8 + 0.2
    ints = ints.clamp(0.2 , 2.0)
    return ints
