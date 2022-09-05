# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 20:50
# @Author  : CMM
# @Site    : 
# @File    : debug.py
# @Software: PyCharm
import os

import torch
from base import get_root_path


def array_detail(*datas, display_data=True):
    for data in datas:
        print("#------------------------------------------")
        if display_data:
            print(data)
        print("数据类型：", type(data))  # 打印数组数据类型
        if 'list' in str(type(data)):
            for i, d in enumerate(data):
                print(f"#----------------第{i}个数据--------------------")
                print("数据类型：", type(d))
                print("数据中元素数据类型：", d.dtype)  # 打印数组元素数据类型
                print("数据形状：", d.shape)  # 打印数组形状
                print("数据的维度数目：", d.ndim)  # 打印数组的维度数目
                if d.ndim > 1 and display_data:
                    print("数据中第一个数据：", d[0])
                print("数据中的最大值：", torch.min(d))
                print("数据中的最小值：", torch.max(d))

        if 'Tensor' in str(type(data)):
            print("数据中元素数据类型：", data.dtype)  # 打印数组元素数据类型
            print("数据形状：", data.shape)  # 打印数组形状
            print("数据的维度数目：", data.ndim)  # 打印数组的维度数目
            if data.ndim > 1 and display_data:
                print("数据中第一个数据：", data[0])
            print("数据中的最大值：", torch.min(data))
            print("数据中的最小值：", torch.max(data))
