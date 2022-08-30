# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 22:20
# @Author  : CMM
# @Site    : 
# @File    : cgan.py
# @Software: PyCharm
import torch
import torch.nn as nn
import config


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(config.N_CLASSES, config.N_CLASSES)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(config.N_Z + config.N_CLASSES, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, config.N_C * config.IMAGE_SIZE * config.IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self, noise, label):
        gen_input = torch.cat((self.label_emb(label), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), config.N_C, config.IMAGE_SIZE, config.IMAGE_SIZE)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(config.N_CLASSES, config.N_CLASSES)
        self.model = nn.Sequential(
            nn.Linear(config.N_CLASSES + config.N_C * config.IMAGE_SIZE * config.IMAGE_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)
        )

    def forward(self, img, labels):
        disc_input = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(disc_input)
        return validity
