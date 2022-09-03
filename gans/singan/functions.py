# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 23:08
# @Author  : CMM
# @File    : functions.py
# @Software: PyCharm
import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from skimage import io as img
from skimage import color
from torch import nn

from gans.singan.imresize import imresize


def post_config(opt):
    # init fixed parameters
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    if opt.mode == 'SR':
        opt.alpha = 100
    return opt


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def read_image(opt):
    x = img.imread('%s/%s' % (opt.DATA_ROOT, opt.DATASET))
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def adjust_scales2image(real_, opt):
    matplotlib_imshow(real_[0], one_channel=False)
    # size * 0.75^num_scales < 25
    opt.num_scales = math.ceil(
        (math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1
    # print(opt.num_scales)
    # size * 0.75^scale2stop < 250
    scale2stop = math.ceil(
        math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
                 opt.scale_factor_init))
    # print(scale2stop)
    opt.stop_scale = opt.num_scales - scale2stop  # 8
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),
                     1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)
    # print(opt.stop_scale)
    # print(opt.scale1)

    real = imresize(real_, opt.scale1, opt)  # torch.Size([1, 3, 165, 250])
    matplotlib_imshow(real[0], one_channel=False)
    # real.shape[2] * scale_factor^stop_scale = opt.min_size
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])),
                                1 / (opt.stop_scale))  # 0.7898725259253863
    scale2stop = math.ceil(
        math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
                 opt.scale_factor_init))  # 4
    opt.stop_scale = opt.num_scales - scale2stop  # 8
    return real


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # [-1~1]=>[0,1]
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")  # cmap='gray'
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def creat_reals_pyramid(real, reals, opt):
    real = real[:, 0:3, :, :]
    for i in range(0, opt.stop_scale + 1, 1):
        scale = math.pow(opt.scale_factor, opt.stop_scale - i)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    return reals


def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def move_to_gpu(t):
    if torch.cuda.is_available():
        t = t.to(torch.device('cuda'))
    return t


def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        # print(inp.shape)
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp, 0, 1)
    return inp


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    if type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise


def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)
