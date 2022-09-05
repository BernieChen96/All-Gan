# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 09:09
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import torch
import torchvision
import config
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import nn, optim

from gans import functions
from gans.base_trainer import BaseTrainer
from gans.dcgan.dcgan import Generator, Discriminator


class Trainer(BaseTrainer):

    def __init__(self, name='dcgan', config=config):
        super(Trainer, self).__init__(name=name, config=config)
        self.classes = None
        self.dataloader = None
        self.dataset = None
        self.fixed_noise = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.criterion = None
        self.net_D = None
        self.net_G = None
        self.real_label = None
        self.fake_label = None
        self.setup()

    def setup(self):
        # Initialize dataset
        self.dataset, self.dataloader, self.classes = self.get_dataset(config.DATASET)
        # Initialize generator and discriminator
        self.net_G = Generator().to(self.device)
        self.net_G.apply(weights_init)
        if config.NET_G != '':
            self.net_G.load_state_dict(torch.load(config.NET_G))
        print(self.net_G)

        self.net_D = Discriminator().to(self.device)
        self.net_D.apply(weights_init)
        if config.NET_D != '':
            self.net_D.load_state_dict(torch.load(config.NET_D))
        print(self.net_D)

        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(config.BATCH_SIZE, config.N_Z, 1, 1, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        # setup optimizer
        self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=config.LR, betas=(config.BETA_1, 0.999))
        self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=config.LR, betas=(config.BETA_1, 0.999))

        if config.DRY_RUN:
            config.N_EPOCH = 1

    def train(self):
        for epoch in range(config.N_EPOCH):
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.optimizer_D.zero_grad()
                real_cpu = data[0].to(self.device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), self.real_label,
                                   dtype=real_cpu.dtype, device=self.device)

                output = self.net_D(real_cpu)
                err_D_real = self.criterion(output, label)
                err_D_real.backward()
                D_x = output.mean().item()

                # train with fake
                noise = torch.randn(batch_size, config.N_Z, 1, 1, device=self.device)
                fake = self.net_G(noise)
                label.fill_(self.fake_label)
                output = self.net_D(fake.detach())
                err_D_fake = self.criterion(output, label)
                err_D_fake.backward()
                D_G_z1 = output.mean().item()
                err_D = err_D_real + err_D_fake
                self.optimizer_D.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizer_G.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                output = self.net_D(fake)
                err_G = self.criterion(output, label)
                err_G.backward()
                D_G_z2 = output.mean().item()
                self.optimizer_G.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, config.N_EPOCH, i, len(self.dataloader),
                         err_D.item(), err_G.item(), D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    # create grid of images
                    grid = torchvision.utils.make_grid(fake.detach())
                    # show images
                    functions.matplotlib_imshow(grid, one_channel=1)
                    # write to tensorboard
                    self.writer.add_image('gen_images', grid, global_step=i)

                    vutils.save_image(real_cpu,
                                      '%s/sample/real_samples.png' % config.OUT_PATH,
                                      normalize=True)
                    fake = self.net_G(self.fixed_noise)
                    vutils.save_image(fake.detach(),
                                      '%s/sample/fake_samples_epoch_%03d.png' % (config.OUT_PATH, epoch),
                                      normalize=True)

                if config.DRY_RUN:
                    break
            # do checkpointing
            torch.save(self.net_G.state_dict(), '%s/checkpoint/netG_epoch_%d.pth' % (config.OUT_PATH, epoch))
            torch.save(self.net_D.state_dict(), '%s/checkpoint/netD_epoch_%d.pth' % (config.OUT_PATH, epoch))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
