# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 08:52
# @Author  : CMM
# @Site    : 
# @File    : config.py
# @Software: PyCharm
import os

from base import get_root_path

# cifar10 | lsun | mnist |imagenet | folder | lfw | fake
DATASET = 'mnist'
# path to dataset
DATA_ROOT = os.path.join(get_root_path(), 'data')
# number of data loading workers'
WORKERS = 0

# input batch size
BATCH_SIZE = 64
# the height / width of the input image to network
IMAGE_SIZE = 64
# channels
N_C = 1
# number of epochs to train for
N_EPOCH = 25
# learning rate, default=0.0002
LR = 0.0002
# beta1 for adam. default=0.5
BETA_1 = 0.5
# size of the latent z vector
N_Z = 100
# 生成器和判别器的维度基数
N_G_F = 64
N_D_F = 64
# 截断值
CLIP_VALUE = 0.01
# 生成器和判别器训练比例
N_CRITIC = 1

# check a single training cycle works
DRY_RUN = False
# number of GPUs to use
N_GPU = 1
# path to netG (to continue training)
NET_G = ''
# path to netD (to continue training)
NET_D = ''
# folder to output images and model checkpoints
OUT_PATH = '.'
# manual seed
MANUAL_SEED = 42
# comma separated list of classes for the lsun data set
classes = 'bedroot'
