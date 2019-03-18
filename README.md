# SDPS-Net
**[SDPS-Net: Self-calibrating Deep Photometric Stereo Networks, CVPR 2019 (Oral)](http://guanyingc.github.io/SDPS-Net/)**.
<br>
[Guanying Chen](http://www.gychen.org), [Kai Han](http://www.hankai.org/), [Boxin Shi](http://alumni.media.mit.edu/~shiboxin/), [Yasuyuki Matsushita](http://www-infobiz.ist.osaka-u.ac.jp/en/member/matsushita/), [Kwan-Yee K. Wong](http://i.cs.hku.hk/~kykwong/)
<br>

This paper addresses the problem of learning based uncalibrated photometric stereo for non-Lambertian surface.
<br>
<p align="center">
    <img src='data/images/buddha.gif' height="250" >
    <img src='data/images/GT.png' height="250" >
</p>

### Dependencies
SDPS-Net is implemented in [PyTorch](https://pytorch.org/) and tested with Ubuntu (14.04 and 16.04), please install PyTorch first following the official instruction. 

- Python 2.7 
- PyTorch (version = 0.40)
- torchvision
- CUDA-8.0/9.0  
- numpy
- scipy
- scikit-image 

You are highly recommended to use Anaconda and create a new environment to run this code.
```shell
# Create a new python2.7 environment named py2.7
conda create -n py2.7 python=2.7

# Activate the created environment
source activate py2.7

# Example commands for installing the dependencies 
conda install pytorch=0.4.1 cuda80 -c pytorch
conda install torchvision -c pytorch
conda install -c anaconda scipy 
conda install -c anaconda scikit-image 

# Download this code
git clone https://github.com/guanyingc/SDPS-Net.git
cd SDPS-Net
```
## Overview:
We provide:
- Trained models
    - LCNet for lighting calibration from input images
    - NENet for normal estimation from input images and estimated lightings.
- Code to test on DiLiGenT main dataset
- Full code to train a new model, including codes for debugging, visualization and logging.

## Testing
#### Download the trained models
```
sh scripts/download_pretrained_models.sh
```

#### Test on the DiLiGenT main dataset
```shell
# Prepare the DiLiGenT main dataset
sh scripts/prepare_diligent_dataset.sh
# This command will first download and unzip the DiLiGenT dataset, and then centered crop 
# the original images based on the object mask with a margin size of 15 pixels.

# Test SDPS-Net on DiLiGenT main dataset using all of the 96 image
CUDA_VISIBLE_DEVICES=0 python eval/run_stage2.py --retrain data/models/LCNet_CVPR2019.pth.tar --retrain_s2 data/models/NENet_CVPR2019.pth.tar
```

## Training
We adopted the publicly available synthetic [PS Blobby and Sculpture datasets](https://github.com/guanyingc/PS-FCN) for training.
To train a new SDPS-Net model, you have to follow the following steps:
#### Download the training data
```shell
# The total size of the zipped synthetic datasets is 4.7+19=23.7 GB 
# and it takes some times to download and unzip the datasets.
sh scripts/download_synthetic_datasets.sh
```

#### First stage: run `main_stage1.py` to train Light Calibration Network (LCNet)
```shell
# Train LCNet on synthetic datasets using 32 input images
CUDA_VISIBLE_DEVICES=0 python main_stage1.py --in_img_num 32
# Please refer to options/base_opt.py and options/stage1_opt.py for more options

# You can find checkpoints and results in data/logdir/
# It takes about 20 hours to train LCNet on a single Titan X Pascal GPU.
```
#### Second stage: run `main_stage2.py` to train Normal Estimation Network (NENet)
```shell
# Train NENet on synthetic datasets using 32 input images
CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num 32 --retrain data/logdir/path/to/checkpointDirOfLCNet/checkpoint20.pth.tar
# Please refer to options/base_opt.py and options/stage2_opt.py for more options

# You can find checkpoints and results in data/logdir/
# It takes about 26 hours to train LCNet on a single Titan X Pascal GPU.
```

## FAQ

#### Q1: How to test SDPS-Net on other dataset?
- You have to implement a customized Dataset class to load your data, which should not be difficult. Please refer to `datasets/UPS_DiLiGenT_main.py` for an example that loads the DiLiGenT main dataset. Precomputed results on DiLiGenT main dataset, Gourd\&Apple dataset, Light Stage Dataset and Synthetic Test dataset are available upon request.

#### Q2: What should I do if I have problem in running your code?
- Please create an issue if you encounter errors when trying to run the code. Please also feel free to submit a bug report.

## Citation
If you find this code or the provided models useful in your research, please consider cite: 
```
@inproceedings{chen2019SDPS_Net,
  title={SDPS-Net: Self-calibrating Deep Photometric Stereo Networks},
  author={Chen, Guanying and Han, Kai and Shi, Boxin and Matsushita, Yasuyuki and Wong, Kwan-Yee K.},
  booktitle={CVPR},
  year={2019}
}
```
