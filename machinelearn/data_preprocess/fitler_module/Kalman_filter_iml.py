#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : Kalman_filter_iml.py
@Time    : 2020/9/9 10:48
@desc	 : 卡尔曼滤波实现
'''

from math import *


def f(mu, sigma2, x):
    return 1 / sqrt(2. * pi * sigma2) * exp(-.5 * (x - mu) ** 2 / sigma2)


def update(mean1, var1, mean2, var2):
    """
    New Mean and Variance
    :param mean1:
    :param var1:
    :param mean2:
    :param var2:
    :return:
    """
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1 / (1 / var1 + 1 / var2)
    return [new_mean, new_var]


# 卡尔曼过滤器，即使没有测量物体的速度，也可以通过物体的位置变化，得到物体的速度，
# 进而预测追踪的物体以该速度出现的下一个位置。
def predict(mean1, var1, mean2, var2):
    """
    Predict Function
    :param mean1:
    :param var1:
    :param mean2:
    :param var2:
    :return:
    """
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]


measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

# Please print out ONLY the final values of the mean
# and the variance in a list [mu, sig].

# Insert code here
for n in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[n], measurement_sig)
    print('update:', [mu, sig])
    [mu, sig] = predict(mu, sig, motion[n], motion_sig)
    print('predict:', [mu, sig])

print([mu, sig])
