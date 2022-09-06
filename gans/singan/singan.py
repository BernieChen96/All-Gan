# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 22:57
# @Author  : CMM
# @File    : singan_bak.py
# @Software: PyCharm
import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, num_filters, num_layer=5, channels=3, kernel_size=3, padding=0, stride=1):
        super(Generator, self).__init__()
        self.head = ConvBlock(channels, num_filters, kernel_size, padding, stride)
        self.padding = nn.ZeroPad2d(int(((kernel_size - 1) * num_layer) / 2))  # 5
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            block = ConvBlock(num_filters, num_filters, kernel_size, padding, stride)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(num_filters, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )

    def forward(self, prev, noise):
        prev_pad = self.padding(prev)
        noise_pad = self.padding(noise)
        x = self.head(torch.add(prev_pad, noise_pad))
        x = self.body(x)
        x = self.tail(x)
        return torch.add(x, prev)


class Discriminator(nn.Module):
    def __init__(self, num_filters, num_layer=5, channels=3, kernel_size=3, padding=0, stride=1):
        super(Discriminator, self).__init__()
        self.head = ConvBlock(channels, num_filters, kernel_size, padding, stride)
        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            block = ConvBlock(num_filters, num_filters, kernel_size, padding, stride)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
