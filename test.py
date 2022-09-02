# -*- coding: utf-8 -*-
# @Time    : 2022/8/27 13:42
# @Author  : CMM
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import unittest


class TestGans(unittest.TestCase):

    def test_get_gpu_info(self):
        from base import get_gpu_info

        get_gpu_info()

    def test_torch_rand(self):
        import torch
        x = torch.rand([16, 1, 1, 1])
        print(x)

    def test_read_image(self):
        from skimage import io as img
        from gans.singan.functions import np2torch, adjust_scales2image, post_config
        from gans.singan.config import get_arguments
        img_ = img.imread("F:\Projects\PythonProjects\DeepLearning\All-Gan\data\SOB_B_A-14-22549AB-40-001.png")
        print(img_.shape)
        parser = get_arguments()
        opt = parser.parse_args()
        x = np2torch(img_, opt)
        # print(x)
        print(x.shape)
        x = x[:, 0:3, :, :]
        print(x.shape)
        post_config(opt)
        adjust_scales2image(x, opt)


if __name__ == '__main__':
    unittest.main()
