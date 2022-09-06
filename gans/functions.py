# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 07:46
# @Author  : CMM
# @Site    : 
# @File    : functions.py
# @Software: PyCharm
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms


def matplotlib_imshow(img, one_channel=False, title=None):
    if img.ndim > 3:
        img = img[0]
    if one_channel:
        img = img.mean(dim=0)
    # [-1~1]=>[0,1]
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")  # cmap='gray'
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()



def imresize(image, h, w):
    resize = transforms.Resize([math.ceil(h), math.ceil(w)])
    image_resized = resize(image)
    return image_resized


def compute_gradient_penalty(D, real_samples, fake_samples, LAMBDA, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)  # 这里如果最终的样本不足，无法自动broadcast，需要将维度填充满
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.pow((gradients.norm(2, dim=1) - 1), 2).mean() * LAMBDA
    return gradient_penalty
