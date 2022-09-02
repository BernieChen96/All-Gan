# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 22:58
# @Author  : CMM
# @File    : trainer.py
# @Software: PyCharm
import math
import os

import torch
from matplotlib import pyplot as plt

from gans.base_trainer import BaseTrainer
from gans.singan import functions
from gans.singan.config import get_arguments
from gans.singan.imresize import imresize
import gans.singan.singan as models
import base


class Trainer(BaseTrainer):

    def __init__(self, opt, name='singan'):
        super(Trainer, self).__init__(name=name, config=opt)
        self.opt = opt
        Gs = []
        Zs = []
        reals = []
        NoiseAmp = []
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        print(real.shape)
        self.train(opt, Gs, Zs, reals, NoiseAmp)

    def setup(self):
        # Initialize generator and discriminator

        pass

    def train(self, opt, Gs, Zs, reals, NoiseAmp):
        real_ = functions.read_image(opt)
        in_s = 0
        scale_num = 0
        real = imresize(real_, opt.scale1, opt)
        reals = functions.creat_reals_pyramid(real, reals, opt)
        nfc_prev = 0

        while scale_num < opt.stop_scale + 1:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = os.path.join(os.path.join(base.get_root_path(), f'gans/{self.name}'),
                                    'TrainedModels/%s/scale_factor=%f,alpha=%d' % (
                                        opt.DATASET[:-4], opt.scale_factor_init, opt.alpha))
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr = self.init_models(opt)
            scale_num += 1

    def init_models(self, opt):
        # generator initialization:
        netG = models.Generator(opt).to(self.device)
        netG.apply(models.weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        print(netG)
        return netG


if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    print(opt)
    trainer = Trainer(opt)
