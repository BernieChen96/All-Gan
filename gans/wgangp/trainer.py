# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 23:10
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import numpy as np

from gans.base_trainer import BaseTrainer
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.transforms as transforms
import config
import torch
from gans.wgangp.wgangp import Discriminator, Generator


class Trainer(BaseTrainer):
    def __init__(self, name='wgangp', config=config):
        super(Trainer, self).__init__(name=name, config=config)

        self.classes = None
        self.dataloader = None
        self.dataset = None
        self.fixed_noise = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.net_D = None
        self.net_G = None
        self.lambda_gp = None
        self.setup()

    def setup(self):
        # Initialize dataset
        self.dataset, self.dataloader, self.classes = self.get_dataset(config.DATASET)

        self.net_G = Generator().to(self.device)
        self.net_D = Discriminator().to(self.device)
        if config.NET_G != '':
            self.net_G.load_state_dict(torch.load(config.NET_G))
        if config.NET_D != '':
            self.net_D.load_state_dict(torch.load(config.NET_D))

        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), config.LR, betas=[config.BETA_1, 0.999])
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), config.LR, betas=[config.BETA_1, 0.999])

        self.fixed_noise = torch.randn(config.BATCH_SIZE, config.N_Z, device=self.device)

        self.lambda_gp = 10

        if config.DRY_RUN:
            config.N_EPOCH = 1

        # 检查模型
        self.summary_graph(self.net_G, self.fixed_noise)
        fake = self.net_G(self.fixed_noise)
        self.summary_graph(self.net_D, fake.detach())
        # 查看数据
        self.summary_embedding(self.dataset, self.classes)

    def train(self):
        for epoch in range(config.N_EPOCH):
            for i, (imgs, _) in enumerate(self.dataloader):
                real_imgs = imgs.to(self.device)
                batch_size = real_imgs.size(0)
                ############################
                # (1) Update D network: maximize D(x) - D(G(z))) + lambda_gp * ( (||∆x`D(alpha*x + (1-alpha)*G(z))||2范数 -1)2平方 )
                ###########################
                self.optimizer_D.zero_grad()
                noise = torch.randn(batch_size, config.N_Z, device=self.device)
                fake_image = self.net_G(noise)
                fake_output = self.net_D(fake_image.detach())
                real_output = self.net_D(real_imgs)
                gradient_penalty = self.compute_gradient_penalty(self.net_D, real_imgs, fake_image.detach())
                loss_D = -torch.mean(real_output) + torch.mean(fake_output) + self.lambda_gp * gradient_penalty
                loss_D.backward()
                self.optimizer_D.step()
                ############################
                # (2) Update G network: maximize D(G(z)))
                ###########################
                if i % config.N_CRITIC == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------
                    self.optimizer_G.zero_grad()
                    loss_G = -torch.mean(self.net_D(fake_image))
                    loss_G.backward()
                    self.optimizer_G.step()
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                          % (epoch, config.N_EPOCH, i, len(self.dataloader),
                             loss_D.item(), loss_G.item()))
                if i % 100 == 0:
                    fake = self.net_G(self.fixed_noise)
                    self.summary_image(fake, epoch * (int(len(self.dataloader) / 100) * 100) + i, one_channel=True)
                    vutils.save_image(real_imgs,
                                      '%s/sample/real_samples.png' % config.OUT_PATH,
                                      normalize=True)
                    vutils.save_image(fake.detach(),
                                      '%s/sample/fake_samples_epoch_%03d.png' % (config.OUT_PATH, epoch),
                                      normalize=True)
                if config.DRY_RUN:
                    break
            # do checkpointing
            torch.save(self.net_G.state_dict(), '%s/checkpoint/netG_epoch_%d.pth' % (config.OUT_PATH, epoch))
            torch.save(self.net_D.state_dict(), '%s/checkpoint/netD_epoch_%d.pth' % (config.OUT_PATH, epoch))

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)  # 这里如果最终的样本不足，无法自动broadcast，需要将维度填充满
        # Get random interpolation between real and fake samples
        print(real_samples.shape)
        print(fake_samples.shape)
        print(alpha.shape)
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
        gradient_penalty = torch.pow((gradients.norm(2, dim=1) - 1), 2).mean()
        return gradient_penalty


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
