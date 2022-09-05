# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 23:40
# @Author  : CMM
# @Site    : 
# @File    : trainer.py
# @Software: PyCharm
import math

from torchvision import transforms
from gans.singan import args
from gans import functions
from gans.base_trainer import BaseTrainer
import config
from gans.singan.datasets import Dataset, get_loader
from util import debug


class Trainer(BaseTrainer):
    def __init__(self, name='singan'):
        super(Trainer, self).__init__(name=name, config=config)
        self.args = args
        self.debug = config.DEBUG
        self.dataset = None
        self.dataloader = None

        self.setup()

    def setup(self):
        # Initialize dataset
        self.dataset = Dataset(config.DATASET, transform=transforms.Compose([
            transforms.ToTensor(),  # 0-1
            transforms.Normalize((0.5,), (0.5,)),  # -1-1
        ]))
        self.dataloader = get_loader(self.dataset, batch_size=1)
        # Initialize image pyramid
        real_imgs = next(iter(self.dataloader)).to(self.device)
        if self.debug:
            # debug.array_detail(real_imgs)
            functions.matplotlib_imshow(real_imgs, one_channel=False, title='original real image')
        real_imgs = self._adjust_scales(real_imgs)
        reals = self._set_reals(real_imgs)  # create real pyramid
        print(reals['s0'].dtype)

    def train(self):
        pass

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
        reals = {}

        # loop over scales
        for i in range(self.args.stop_scale + 1):
            s = math.pow(self.args.scale_factor, self.args.stop_scale - i)
            h = real.size(2) * s
            w = real.size(3) * s
            tmp_real = functions.imresize(real, h=h, w=w)
            if self.debug:
                functions.matplotlib_imshow(tmp_real,
                                            title=f'{i}th resize {tmp_real.size(3)}*{tmp_real.size(2)} real image')
            reals.update({'s{}'.format(i): tmp_real})

        return reals


if __name__ == '__main__':
    trainer = Trainer()
