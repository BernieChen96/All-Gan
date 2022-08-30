# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 09:18
# @Author  : CMM
# @Site    : 
# @File    : base_trainer.py
# @Software: PyCharm
import os
from abc import abstractmethod, ABC
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.backends import cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from base import get_root_path
from util.ops import array_detail


class BaseTrainer(ABC):
    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def __init__(self, name):
        self.config = self.get_config()
        # 创建输出文件夹
        try:
            os.makedirs(self.config.OUT_PATH)
        except OSError:
            pass

        # 设置随机种子
        if self.config.MANUAL_SEED is None:
            self.config.MANUAL_SEED = random.randint(1, 10000)
        print("Random Seed: ", self.config.MANUAL_SEED)
        random.seed(self.config.MANUAL_SEED)
        torch.manual_seed(self.config.MANUAL_SEED)

        # Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异
        cudnn.benchmark = True

        # dataroot是否存在
        if self.config.DATA_ROOT is None and str(self.config.DATASET).lower() != 'fake':
            raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % self.opt.dataset)

        self.name = name
        self.dataset_name = self.config.DATASET
        self.init_directory('sample', 'checkpoint', 'runs')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.writer = self.init_summary()
        print("当前使用 device：", self.device)

    def get_dataset(self, name):
        dataset = None
        dataloader = None
        classes = None
        if name == 'mnist':
            dataset = dset.MNIST(root=self.config.DATA_ROOT, download=True,
                                 transform=transforms.Compose([
                                     transforms.Resize(self.config.IMAGE_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                 ]))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.config.BATCH_SIZE,
                                                     shuffle=True, num_workers=int(self.config.WORKERS))
            classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        else:
            print("Please set dataset!")
            exit(0)
        return dataset, dataloader, classes

    def init_directory(self, *args):
        root_path = os.path.join(get_root_path(), 'gans')
        dir_path = os.path.join(root_path, self.name)
        for i in args:
            create_path = os.path.join(dir_path, i)
            if not os.path.exists(create_path):
                os.makedirs(create_path)

    def init_summary(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_dir = os.path.join(get_root_path(), 'gans/%s/runs/%s' % (self.name, self.dataset_name))
        print(self.dataset_name)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_writer = SummaryWriter(os.path.join(summary_dir, current_time))
        return summary_writer

    def matplotlib_imshow(self, img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        # [-1~1]=>[0,1]
        img = img / 2 + 0.5  # unnormalize
        npimg = img.cpu().numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")  # cmap='gray'
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def summary_image(self, image, step, one_channel=False):
        # create grid of images
        grid = torchvision.utils.make_grid(image.detach())
        # show images
        self.matplotlib_imshow(grid, one_channel=one_channel)
        # write to tensorboard
        self.writer.add_image('gen_images', grid, global_step=step)

    def summary_graph(self, net, input):
        self.writer.add_graph(net, input)
        self.writer.close()

    def summary_embedding(self, trainset, classes):
        def select_n_random(data, labels, n=16):
            '''
            Selects n random datapoints and their corresponding labels from a dataset
            '''
            assert len(data) == len(labels)

            perm = torch.randperm(len(data))
            return data[perm][:n], labels[perm][:n]

        # select random images and their target indices
        images, labels = select_n_random(trainset.data, trainset.targets)
        # get the class labels for each image
        class_labels = [classes[lab] for lab in labels]
        # log embeddings
        features = images.view(-1, 28 * 28)
        self.writer.add_embedding(features,
                                  metadata=class_labels,
                                  label_img=images.unsqueeze(1))
        self.writer.close()
