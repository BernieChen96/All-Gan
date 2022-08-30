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
        x = torch.rand([16, 1,1,1])
        print(x)


if __name__ == '__main__':
    unittest.main()
