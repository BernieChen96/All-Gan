# -*- coding: utf-8 -*-
# @Time    : 2022/8/28 22:41
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import torch

from gans.base_trainer import BaseTrainer
from gans.cgan.cgan import Generator, Discriminator
import torchvision.utils as vutils
import config


class Trainer(BaseTrainer):
    def __init__(self, name='cgan'):
        super(Trainer, self).__init__(name=name)

        self.dataset = None
        self.dataloader = None
        self.classes = None
        self.fixed_noise = None
        self.fixed_labels = None
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
        self.net_D = Discriminator().to(self.device)
        if config.NET_G != '':
            self.net_G.load_state_dict(torch.load(config.NET_G))
        print(self.net_G)
        if config.NET_D != '':
            self.net_D.load_state_dict(torch.load(config.NET_D))
        print(self.net_D)

        # Loss Function
        self.criterion = torch.nn.MSELoss()

        # Initialize optimizer
        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), config.LR, betas=(config.BETA_1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), config.LR, betas=(config.BETA_1, 0.999))

        self.fixed_noise = torch.randn(config.BATCH_SIZE, config.N_Z, device=self.device)
        self.fixed_labels = torch.arange(0, config.BATCH_SIZE, device=self.device)
        for i in range(len(self.fixed_labels)):
            self.fixed_labels[i] = self.fixed_labels[i] % config.N_CLASSES
        self.real_label = 1
        self.fake_label = 0

        if config.DRY_RUN:
            config.N_EPOCH = 1
        # 检查模型
        self.summary_graph(self.net_G, (self.fixed_noise, self.fixed_labels))
        fake = self.net_G(self.fixed_noise, self.fixed_labels)
        self.summary_graph(self.net_D, (fake.detach(), self.fixed_labels))
        # 查看数据
        self.summary_embedding(self.dataset, self.classes)

    def get_config(self):
        return config

    def train(self):
        for epoch in range(config.N_EPOCH):
            for i, (imgs, labels) in enumerate(self.dataloader):
                # Adversarial ground truths
                real_imgs = imgs.to(self.device)
                real_labels = labels.to(self.device)
                batch_size = real_imgs.shape[0]
                valid_label = torch.full((batch_size, 1), self.real_label,
                                         dtype=real_imgs.dtype, device=self.device)
                fake_label = torch.full((batch_size, 1), self.fake_label, dtype=real_imgs.dtype, device=self.device)
                ############################
                # (1) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizer_G.zero_grad()
                noise = torch.randn(batch_size, config.N_Z, device=self.device)
                gen_labels = torch.randint(0, config.N_CLASSES, (batch_size,), device=self.device)
                fake_image = self.net_G(noise, gen_labels)
                output = self.net_D(fake_image, gen_labels)
                err_G = self.criterion(output, valid_label)
                err_G.backward()
                self.optimizer_G.step()
                ############################
                # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.optimizer_D.zero_grad()
                # train with real
                output = self.net_D(real_imgs, real_labels)
                err_D_real = self.criterion(output, valid_label)
                err_D_real.backward()
                # train with fake
                output = self.net_D(fake_image.detach(), gen_labels.detach())
                err_D_fake = self.criterion(output, fake_label)
                err_D_fake.backward()
                err_D = err_D_real + err_D_fake
                self.optimizer_D.step()
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, config.N_EPOCH, i, len(self.dataloader),
                         err_D.item(), err_G.item()))

                if i % 100 == 0:
                    fake = self.net_G(self.fixed_noise, self.fixed_labels)
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


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
