import torch
import random
import numpy as np
from skimage.transform import resize
random.seed(0)
np.random.seed(0)

def arrayToTensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()

def normalToMask(normal, thres=1e-2):
    """
    Due to the numerical precision of uint8, [0, 0, 0] will save as [127, 127, 127] in gt normal,
    When we load the data and rescale normal by N / 255 * 2 - 1, [127, 127, 127] becomes 
    [-0.003927, -0.003927, -0.003927]
    """
    mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
    return mask

def imgSizeToFactorOfK(img, k):
    if img.shape[0] % k == 0 and img.shape[1] % k == 0:
        return img
    pad_h, pad_w = k - img.shape[0] % k, k - img.shape[1] % k
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0,0)), 
            'constant', constant_values=((0,0),(0,0),(0,0)))
    return img

def randomCrop(inputs, target, size):
    h, w, _ = inputs.shape
    c_h, c_w = size
    if h == c_h and w == c_w:
        return inputs, target
    x1 = random.randint(0, w - c_w)
    y1 = random.randint(0, h - c_h)
    inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
    target = target[y1: y1 + c_h, x1: x1 + c_w]
    return inputs, target

def centerCrop(inputs, size):
    h, w, _ = inputs.shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        x1 = int(w / 2 - c_w / 2)
        y1 = int(h / 2 - c_h / 2)
        inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
    return inputs

def rescale(inputs, target, size):
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = resize(inputs, size, order=1, mode='reflect')
        target = resize(target, size, order=1, mode='reflect')
    return inputs, target

def rescaleSingle(inputs, size, order=1):
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        inputs = resize(inputs, size, order=order, mode='reflect')
    return inputs

def randomNoiseAug(inputs, noise_level=0.05):
    noise = np.random.random(inputs.shape)
    noise = (noise - 0.5) * noise_level
    inputs += noise
    return inputs

def getIntensity(num):
    intensity = np.random.random((num, 1)) * 1.8 + 0.2
    color = np.ones((1, 3)) # Uniform color
    intens = (intensity.repeat(3, 1) * color)
    return intens
