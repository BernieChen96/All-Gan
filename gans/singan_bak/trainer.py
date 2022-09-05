# -*- coding: utf-8 -*-
# @Time    : 2022/9/1 22:58
# @Author  : CMM
# @File    : trainer.py
# @Software: PyCharm
import math
import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from gans.base_trainer import BaseTrainer
from gans.singan_bak import functions
from gans.singan_bak.config import get_arguments
from gans.singan_bak.imresize import imresize
import gans.singan_bak.singan as models
import base


class Trainer(BaseTrainer):

    def __init__(self, opt, name='singan_bak'):
        super(Trainer, self).__init__(name=name, config=opt)
        self.opt = opt
        Gs = []
        Zs = []
        reals = []
        NoiseAmp = []
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        print('原图片调整后的shape：', real.shape)
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

            print(f"初始化第{scale_num}层网络......")
            G_curr, D_curr = self.init_models(opt)
            print(f"初始化第{scale_num}层网络完毕......")
            # 加载前一层网络学习到的参数
            if (nfc_prev == opt.nfc):
                G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
                D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))

            # z_curr, in_s, G_curr = self.train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)
            print(reals[scale_num].shape)
            scale_num += 1

    def train_single_scale(self, netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):

        real = reals[len(Gs)]
        print("当前尺度下图片shape为：", real.shape)
        opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)
        # 感受野 11 = 3+2*4
        opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)  # 5
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

        m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))

        alpha = opt.alpha

        fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=self.device)
        z_opt = torch.full(fixed_noise.shape, 0, device=self.device)  # (1,3,25,28)
        z_opt = m_noise(z_opt)  # (1,3,35,48)

        # setup optimizer
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

        errD2plot = []
        errG2plot = []
        D_real2plot = []
        D_fake2plot = []
        z_opt2plot = []

        for epoch in range(opt.niter):
            if not Gs:
                z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], device=self.device)
                z_opt = m_noise(z_opt.expand(1, 3, opt.nzx, opt.nzy))
                noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], device=self.device)
                noise_ = m_noise(noise_.expand(1, 3, opt.nzx, opt.nzy))
            else:
                noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=self.device)
                noise_ = m_noise(noise_)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
                # train with real
                netD.zero_grad()

                output = netD(real).to(self.device)
                # D_real_map = output.detach()
                errD_real = -output.mean()  # -a
                errD_real.backward(retain_graph=True)
                D_x = -errD_real.item()

                # train with fake
                if (j == 0) & (epoch == 0):
                    if not Gs:
                        prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=self.device)
                        in_s = prev
                        prev = m_image(prev)
                        z_prev = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=self.device)
                        z_prev = m_noise(z_prev)
                        opt.noise_amp = 1
                    else:
                        prev = self.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                        prev = m_image(prev)
                        z_prev = self.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init * RMSE
                        z_prev = m_image(z_prev)
                else:
                    prev = self.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    prev = m_image(prev)

                if not Gs:
                    noise = noise_
                else:
                    noise = opt.noise_amp * noise_ + prev

                fake = netG(noise, prev)
                output = netD(fake.detach())
                errD_fake = output.mean()
                errD_fake.backward(retain_graph=True)
                D_G_z = output.mean().item()

                gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, self.device)
                gradient_penalty.backward()

                errD = errD_real + errD_fake + gradient_penalty
                optimizerD.step()

            errD2plot.append(errD.detach())

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(opt.Gsteps):
                netG.zero_grad()
                output = netD(fake)
                # D_fake_map = output.detach()
                errG = -output.mean()
                errG.backward(retain_graph=True)
                if alpha != 0:
                    loss = nn.MSELoss()
                    if opt.mode == 'paint_train':
                        z_prev = functions.quant2centers(z_prev, centers)
                        plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                    Z_opt = opt.noise_amp * z_opt + z_prev
                    rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev), real)
                    rec_loss.backward(retain_graph=True)
                    rec_loss = rec_loss.detach()
                else:
                    Z_opt = z_opt
                    rec_loss = 0

                optimizerG.step()

            errG2plot.append(errG.detach() + rec_loss)
            D_real2plot.append(D_x)
            D_fake2plot.append(D_G_z)
            z_opt2plot.append(rec_loss)

            if epoch % 25 == 0 or epoch == (opt.niter - 1):
                print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

            if epoch % 500 == 0 or epoch == (opt.niter - 1):
                plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt).png' % (opt.outf),
                           functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
                # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
                # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
                # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
                # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
                # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
                # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

                torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

            schedulerD.step()
            schedulerG.step()

        functions.save_networks(netG, netD, z_opt, opt)
        return z_opt, in_s, netG

    def init_models(self, opt):
        # generator initialization:
        netG = models.Generator(opt).to(self.device)
        netG.apply(models.weights_init)
        if opt.netG != '':
            netG.load_state_dict(torch.load(opt.netG))
        print(netG)

        # discriminator initialization:
        netD = models.WDiscriminator(opt).to(self.device)
        netD.apply(models.weights_init)
        if opt.netD != '':
            netD.load_state_dict(torch.load(opt.netD))
        print(netD)
        return netG, netD

    def draw_concat(self, Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
        G_z = in_s
        if len(Gs) > 0:
            if mode == 'rand':
                count = 0
                pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
                if opt.mode == 'animation_train':
                    pad_noise = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                    if count == 0:
                        z = functions.generate_noise(
                            [1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                        z = z.expand(1, 3, z.shape[2], z.shape[3])
                    else:
                        z = functions.generate_noise(
                            [opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                            device=opt.device)
                    z = m_noise(z)
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                    G_z = m_image(G_z)
                    z_in = noise_amp * z + G_z
                    G_z = G(z_in.detach(), G_z)
                    G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    count += 1
            if mode == 'rec':
                count = 0
                for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                    G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                    G_z = m_image(G_z)
                    z_in = noise_amp * Z_opt + G_z
                    G_z = G(z_in.detach(), G_z)
                    G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    # if count != (len(Gs)-1):
                    #    G_z = m_image(G_z)
                    count += 1
        return G_z


if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    print(opt)
    trainer = Trainer(opt)
