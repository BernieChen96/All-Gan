# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 07:58
# @Author  : CMM
# @Site    : 
# @File    : datasets.py
# @Software: PyCharm
import os

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms
from PIL import Image

from base import get_root_path


class Dataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(Dataset, self).__init__()
        root = os.path.join(get_root_path(), 'data')
        self.img_path = os.path.join(root, img_path)
        self.transform = transform

    def __getitem__(self, index) -> T_co:
        img = Image.open(self.img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        # img = (img - 0.5) * 2
        return img

    def __len__(self):
        return 1


def get_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
# class Dataset(Dataset):
#     def __init__(self, root='', crop_size=0):
#         self.root = root
#         self.crop_size = crop_size
#         self._init()
#
#     def _init(self):
#         # to tensor
#         self.to_tensor = transforms.ToTensor()
#
#         # open image
#         image = Image.open(self.root).convert('RGB')
#         self.image = self.to_tensor(image).unsqueeze(dim=0)
#         self.image = (self.image - 0.5) * 2
#
#         # set from outside
#         self.reals = None
#         self.noises = None
#         self.amps = None
#
#     def _get_augment_params(self, size):
#         # position
#         w_size, h_size = size
#         x = random.randint(0, max(0, w_size - self.crop_size))
#         y = random.randint(0, max(0, h_size - self.crop_size))
#
#         # flip
#         flip = random.random() > 0.5
#         return {'pos': (x, y), 'flip': flip}
#
#     def _augment(self, image, aug_params, scale=1):
#         x, y = aug_params['pos']
#         image = image[:, round(x * scale):(round(x * scale) + round(self.crop_size * scale)),
#                 round(y * scale):(round(y * scale) + round(self.crop_size * scale))]
#         if aug_params['flip']:
#             image = image.flip(-1)
#         return image
#
#     def __getitem__(self, index):
#         amps = self.amps
#
#         # cropping
#         if self.crop_size:
#             reals, noises = {}, {}
#             aug_params = self._get_augment_params(self.image.size()[-2:])
#
#             for key in self.reals.keys():
#                 scale = self.reals[key].size(-1) / float(self.image.size(-1))
#                 reals.update({key: self._augment(self.reals[key].clone(), aug_params, scale)})
#                 noises.update({key: self._augment(self.noises[key].clone(), aug_params, scale)})
#
#         # full size
#         else:
#             reals = self.reals  # TODO: clone when crop
#             noises = self.noises  # TODO: clone when crop
#
#         return {'reals': reals, 'noises': noises, 'amps': amps}
#
#     def __len__(self):
#         return self.batch_size
