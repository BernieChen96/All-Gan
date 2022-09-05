# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 10:51
# @Author  : CMM
# @Site    : 
# @File    : args.py
# @Software: PyCharm

# 初始化定义
min_size = 25  # 图片最小大小
max_size = 250
scale_factor_init = 0.75  # 图片金字塔缩放比例
# 过程中定义
scale_factor = None
num_scales = None
scale_to_stop = None
stop_scale = None  # 金字塔层数
scale_one = None
