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
min_filters = 32
lr = 0.0005  # 学习率
gen_beta1 = 0.5
gen_beta2 = 0.999
disc_beta1 = 0.5
disc_beta2 = 0.999
mutil = 1
step_size = 2000 * mutil
gamma = 0.1
lambda_grad = 0.1  # gp权重
alpha = 10  # 重建损失权重

num_iters = 4000 * mutil
# 过程中定义
scale_factor = None
num_scales = None
scale_to_stop = None
stop_scale = None  # 金字塔层数
scale_one = None
