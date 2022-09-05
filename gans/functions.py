# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 07:46
# @Author  : CMM
# @Site    : 
# @File    : functions.py
# @Software: PyCharm
import math

import numpy as np
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
