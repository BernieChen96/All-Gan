# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 23:40
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import math
import os.path
import time

import torch
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

import config
from gans import functions
from gans.base_trainer import BaseTrainer
from gans.singan import args
from gans.singan.datasets import Dataset, get_loader
from gans.singan.singan import Generator, Discriminator


class Trainer(BaseTrainer):
    def __init__(self, name='singan'):
        super(Trainer, self).__init__(name=name, config=config)

        self.args = args
        self.debug = config.DEBUG
        self.dataset = None
        self.dataloader = None
        self.scale = None  # 当前正在执行的尺度
        self.reals = None  # create real pyramid
        self.noise = None  # create noise pyramid
        self.num_filters = None  # 生成器和判别器所需的核数
        self.generators = None
        self.discriminator = None
        self.NoiseAmp = None
        self.Z_fixed = None
        self.noise_amp_init = None
        self.reconstruction = None
        self.setup()

    def setup(self):
        # Initialize dataset
        print("初始化数据集......")
        self.dataset = Dataset(config.DATASET, transform=transforms.Compose([
            transforms.ToTensor(),  # 0-1
            transforms.Normalize((0.5,), (0.5,)),  # -1-1
        ]))
        self.dataloader = get_loader(self.dataset, batch_size=1)
        print("数据集初始化完毕......")

        # Initialize image pyramid
        print("初始化图像......")
        real_imgs = next(iter(self.dataloader))
        if self.debug:
            # debug.array_detail(real_imgs)
            functions.matplotlib_imshow(real_imgs, one_channel=False, title='original real image')
        real_imgs = self._adjust_scales(real_imgs)
        self.reals = self._set_reals(real_imgs)  # create real pyramid
        print("图像初始化完毕......")

        # Initialize D and G
        print("初始化模型......")
        self.num_filters = [self.args.min_filters * pow(2, (scale // 4)) for scale in range(self.args.stop_scale + 1)]
        self._build_model()
        print("模型初始化完毕......")

        #  噪声缩放初始化
        self.noise_amp_init = 0.1

        # criterion
        self.reconstruction = torch.nn.MSELoss()

    def train(self):
        self.Z_fixed = []
        self.NoiseAmp = []
        noise_amp = self.noise_amp_init  # 定义一个
        print("开始训练......")
        # iterate scales
        for scale in range(self.args.stop_scale + 1):
            print(f"对于第{scale}尺度进行训练")
            start = time.perf_counter()
            real_img = self.reals[scale].to(self.device)
            generator = self.generators[scale]
            discriminator = self.discriminators[scale]
            if not self._load_model(scale, generator, discriminator):
                print(f"第{scale}尺度模型权重不存在")
                jump = False
                if scale > 0:
                    # 除去第一个尺度，都加载前面的模型
                    if self._init_from_previous_model(scale, generator, discriminator):
                        print(f"加载第{scale - 1}尺度模型成功......")
                    else:
                        print(f"加载第{scale - 1}尺度模型失败......")
            else:
                print(f"加载第{scale}尺度模型成功......")
                jump = True
            # initialize optimizer
            g_optimizer = torch.optim.Adam(generator.parameters(), lr=self.args.lr,
                                           betas=(self.args.gen_beta1, self.args.gen_beta2))
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr,
                                           betas=(self.args.disc_beta1, self.args.disc_beta2))

            # initialize scheduler
            g_scheduler = StepLR(g_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
            d_scheduler = StepLR(d_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

            prev_rec = torch.zeros_like(real_img, device=self.device)
            if self.debug:
                print("DEBUG模式......")
                print("展示图片......")
                functions.matplotlib_imshow(real_img,
                                            title=f'{scale}th resize {real_img.size(3)}*{real_img.size(2)} real image')
                print("测试模型......")
                print("D(0): ", discriminator(prev_rec).shape)
                print("G(0,0): ", generator(prev_rec, prev_rec).shape)
            for step in range(self.args.num_iters):
                z_fixed, prev_rec, noise_amp, metrics = self.train_step(real_img, prev_rec, noise_amp, scale, step,
                                                                        discriminator, generator, d_optimizer,
                                                                        g_optimizer)
                d_scheduler.step()
                g_scheduler.step()

                if step % 20 == 0:
                    # print(f"z_fixed:{z_fixed}, prev_rec:{prev_rec}, noise_amp:{noise_amp}, metrics:{metrics}")
                    fake = generator(prev_rec, z_fixed)
                    self.summary_image(fake, scale * self.args.num_iters + step, f'{scale}th image', one_channel=False,
                                       debug=False)
                    # errD.item(), errG.item(), rec_loss.item()
                    self.writer.add_scalar(f"{scale}th errD", metrics[0], step)
                    self.writer.add_scalar(f"{scale}th errG", metrics[1], step)
                    self.writer.add_scalar(f"{scale}th rec_loss", metrics[2], step)

                if step + 1 % 1000 == 0:
                    self._save_model(scale, step, generator, discriminator)
                if jump:
                    break

            if self.debug:
                functions.matplotlib_imshow(generator(prev_rec, z_fixed).detach())
            self.Z_fixed.append(z_fixed)
            self.NoiseAmp.append(noise_amp)

            print(f'Time taken for scale {scale} is {time.perf_counter() - start:.2f} sec\n')

        if config.DRY_RUN:
            exit(0)
        print("训练结束......")

    def train_step(self, real, prev_rec, noise_amp, scale, step, discriminator, generator, d_optimizer, g_optimizer):
        z_rand = torch.randn(real.shape, device=self.device)
        if scale == 0:
            z_rec = torch.randn(real.shape, device=self.device)
        else:
            z_rec = torch.zeros_like(real, device=self.device)
        for i in range(6):
            if i == 0 and step == 0:
                if scale == 0:
                    """ Coarsest scale is purely generative """
                    prev_rand = torch.zeros_like(real, device=self.device)
                    prev_rec = torch.zeros_like(real, device=self.device)
                    noise_amp = 1.0
                else:
                    """ Finer scale takes noise and image generated from previous scale as input """
                    prev_rand = self.generate_from_coarsest(scale, 'rand')
                    prev_rec = self.generate_from_coarsest(scale, 'rec')
                    """ Compute the standard deviation of noise """
                    RMSE = torch.sqrt(torch.mean(torch.square(real - prev_rec)))
                    noise_amp = self.noise_amp_init * RMSE
            else:
                prev_rand = self.generate_from_coarsest(scale, 'rand')

            Z_rand = z_rand if scale == 0 else noise_amp * z_rand
            Z_rec = noise_amp * z_rec

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################

            # train with real
            d_optimizer.zero_grad()
            real_output = discriminator(real)
            # train with fake
            fake_rand = generator(prev_rand, Z_rand)
            fake_output = discriminator(fake_rand.detach())
            # gp
            gradient_penalty = functions.compute_gradient_penalty(discriminator, real.detach(), fake_rand.detach(),
                                                                  self.args.lambda_grad,
                                                                  self.device)
            errD = -torch.mean(real_output) + torch.mean(fake_output) + gradient_penalty
            errD.backward(retain_graph=True)
            d_optimizer.step()
            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################
            g_optimizer.zero_grad()
            gen_loss = -torch.mean(discriminator(fake_rand))
            rec_loss = 0
            if self.args.alpha != 0:
                fake_rec = generator(prev_rec, Z_rec)
                rec_loss = self.args.alpha * self.reconstruction(fake_rec, real)
            errG = gen_loss + rec_loss
            errG.backward(retain_graph=True)
            g_optimizer.step()
        metrics = (errD.item(), errG.item(), rec_loss.item())
        return z_rec, prev_rec, noise_amp, metrics

    def generate_from_coarsest(self, scale, mode='rand'):
        """ Use random/fixed noise to generate from coarsest scale"""
        fake = torch.zeros_like(self.reals[0], device=self.device)
        if scale > 0:
            if mode == 'rand':
                for i in range(scale):
                    z_rand = torch.randn(self.reals[i].shape, device=self.device)
                    z_rand = self.NoiseAmp[i] * z_rand
                    fake = self.generators[i](fake, z_rand)
                    fake = functions.imresize(fake, self.reals[i + 1].size(2), self.reals[i + 1].size(3))

            if mode == 'rec':
                for i in range(scale):
                    z_fixed = self.NoiseAmp[i] * self.Z_fixed[i]
                    fake = self.generators[i](fake, z_fixed)
                    fake = functions.imresize(fake, self.reals[i + 1].size(2), self.reals[i + 1].size(3))
        return fake

    def _init_from_previous_model(self, scale, generator, discriminator):
        """ Initialize current model from the previous trained model """
        if self.num_filters[scale] == self.num_filters[scale - 1]:
            return self._load_model(scale - 1, generator, discriminator)

    def _load_model(self, scale, generator, discriminator):
        print(f"加载第{scale}尺度模型......")
        if os.path.exists('%s/checkpoint/%s/netG_step_%d.pth' % (self.get_dir_path(), scale, self.args.num_iters - 1)):
            generator.load_state_dict(torch.load(
                '%s/checkpoint/%s/netG_step_%d.pth' % (self.get_dir_path(), scale, self.args.num_iters - 1)))
            discriminator.load_state_dict(torch.load(
                '%s/checkpoint/%s/netD_step_%d.pth' % (self.get_dir_path(), scale, self.args.num_iters - 1)))
            return True
        else:
            return False

    def _save_model(self, scale, step, generator, discriminator):
        dir_path = '%s/checkpoint/%s/' % (self.get_dir_path(), scale)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(generator.state_dict(),
                   '%s/checkpoint/%s/netG_step_%d.pth' % (self.get_dir_path(), scale, step))
        torch.save(discriminator.state_dict(),
                   '%s/checkpoint/%s/netD_step_%d.pth' % (self.get_dir_path(), scale, step))

    def _build_model(self):
        """ Build initial model """
        self.generators = []
        self.discriminators = []
        for scale in range(self.args.stop_scale + 1):
            generator = Generator(num_filters=self.num_filters[scale]).to(self.device)
            discriminator = Discriminator(num_filters=self.num_filters[scale]).to(self.device)
            self.generators.append(generator)
            self.discriminators.append(discriminator)
            if self.debug:
                print(self.generators[scale])
                print(self.discriminators[scale])

    def _adjust_scales(self, image):
        def get_scale_to_stop():
            return math.ceil(math.log(
                min([self.args.max_size, max([image.size(2), image.size(3)])]) / max([image.size(2), image.size(3)]),
                self.args.scale_factor_init))

        self.args.num_scales = math.ceil((math.log(
            math.pow(self.args.min_size / (min(image.size(2), image.size(3))), 1), self.args.scale_factor_init))) + 1
        self.args.scale_to_stop = get_scale_to_stop()
        self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop
        self.args.scale_one = min(self.args.max_size / max([image.size(2), image.size(3)]), 1)

        image_resized = functions.imresize(image=image, h=image.size(2) * self.args.scale_one,
                                           w=image.size(3) * self.args.scale_one)
        if self.debug:
            # debug.array_detail(image_resized)
            functions.matplotlib_imshow(image_resized, one_channel=False, title='resize original real image')
        self.args.scale_factor = math.pow(self.args.min_size / (min(image_resized.size(2), image_resized.size(3))),
                                          1 / self.args.stop_scale)
        self.args.scale_to_stop = get_scale_to_stop()
        self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop
        if self.debug:
            print(f"初始缩放因子设定 scale factor init:{self.args.scale_factor_init}\n"
                  f"调整后的缩放因子 scale factor:{self.args.scale_factor}\n"
                  f"scale to stop:{self.args.scale_to_stop}\n"
                  f"stop scale:{self.args.stop_scale}")
        return image_resized

    def _set_reals(self, real):
        reals = []

        # loop over scales
        for i in range(self.args.stop_scale + 1):
            s = math.pow(self.args.scale_factor, self.args.stop_scale - i)
            h = real.size(2) * s
            w = real.size(3) * s
            tmp_real = functions.imresize(real, h=h, w=w)
            # if self.debug:
            #     functions.matplotlib_imshow(tmp_real,
            #                                 title=f'{i}th resize {tmp_real.size(3)}*{tmp_real.size(2)} real image')
            reals.append(tmp_real)

        return reals


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
