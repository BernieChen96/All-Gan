# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 20:23
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import torchvision.utils as vutils
import torchvision.datasets as dset
from gans.base_trainer import BaseTrainer
from gans.wgan.wgan import Generator, Discriminator
import torchvision.transforms as transforms
import torch
import config


class Trainer(BaseTrainer):
    def train(self):
        for epoch in range(config.N_EPOCH):
            for i, (imgs, _) in enumerate(self.dataloader):
                real_imgs = imgs.to(self.device)
                batch_size = real_imgs.size(0)
                ############################
                # (1) Update D network: maximize D(x) - D(G(z)))
                ###########################
                self.optimizer_D.zero_grad()
                noise = torch.randn(batch_size, config.N_Z, device=self.device)
                fake_image = self.net_G(noise)
                fake_output = self.net_D(fake_image.detach())
                real_output = self.net_D(real_imgs)
                loss_D = -torch.mean(real_output) + torch.mean(fake_output)
                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.net_D.parameters():
                    p.data.clamp_(-config.CLIP_VALUE, config.CLIP_VALUE)

                ############################
                # (2) Update G network: maximize D(G(z)))
                ###########################
                if i % config.N_CRITIC == 0:
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

        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(self.net_G.parameters(), config.LR)
        self.optimizer_D = torch.optim.RMSprop(self.net_D.parameters(), config.LR)

        self.fixed_noise = torch.randn(config.BATCH_SIZE, config.N_Z, device=self.device)

        if config.DRY_RUN:
            config.N_EPOCH = 1

        # 检查模型
        self.summary_graph(self.net_G, self.fixed_noise)
        fake = self.net_G(self.fixed_noise)
        self.summary_graph(self.net_D, fake.detach())
        # 查看数据
        self.summary_embedding(self.dataset, self.classes)

    def get_config(self):
        return config

    def __init__(self, name='wgan'):
        super(Trainer, self).__init__(name=name)

        self.classes = None
        self.dataloader = None
        self.dataset = None
        self.fixed_noise = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.net_D = None
        self.net_G = None
        self.setup()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
