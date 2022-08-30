# -*- coding: utf-8 -*-
# @Time    : 2022/8/26 18:56
# @Author  : CMM
# @Site    : 
# @File    : base.py
# @Software: PyCharm
import os

import torch


def get_root_path():
    """
    得到项目根路径
    :return:
    """
    root_path = os.path.join(
        os.path.dirname(__file__),
    )
    return root_path


def get_mnist_data_path():
    return os.path.join(get_root_path(), "data/mnist")


def get_gpu_info() -> str:
    info = ''
    for id in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(id)
        info += f'CUDA:{id} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n'
    return info[:-1]


