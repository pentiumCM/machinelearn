#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : filter_iml.py
@Time    : 2020/9/8 21:44
@desc	 : 滤波算法实现
            参考：https://blog.csdn.net/kengmila9393/article/details/81455165
'''

import scipy.signal as signal
import numpy as np
# import pylab as pl
import matplotlib.pyplot as plt


def ArithmeticAverage(inputs, per):
    """
    算术平均滤波法
    :param inputs:
    :param per:
    :return:
    """
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


def SlidingAverage(inputs, per):
    """
    递推平均滤波法
    :param inputs:
    :param per:
    :return:
    """
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


def MedianAverage(inputs, per):
    """
    中位值平均滤波法
    :param inputs:
    :param per:
    :return:
    """
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
        mean.append(tmp.mean())
    return mean


def AmplitudeLimitingAverage(inputs, per, Amplitude):
    """
    限幅平均滤波法
    Amplitude:	限制最大振幅
    :param inputs:
    :param per:
    :param Amplitude:
    :return:
    """
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0]  # 上一次限幅后结果
    for tmp in inputs:
        for index, newtmp in enumerate(tmp):
            if np.abs(tmpnum - newtmp) > Amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


def FirstOrderLag(inputs, a):
    """
    一阶滞后滤波法
    a: 滞后程度决定因子，0~1
    :param inputs:
    :param a:
    :return:
    """
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs


def WeightBackstepAverage(inputs, per):
    """
    加权递推平均滤波法
    :param inputs:
    :param per:
    :return:
    """
    weight = np.array(range(1, np.shape(inputs)[0] + 1))  # 权值列表
    weight = weight / weight.sum()

    for index, tmp in enumerate(inputs):
        inputs[index] = inputs[index] * weight[index]
    return inputs


def ShakeOff(inputs, N):
    """
    消抖滤波法
    N:	消抖上限
    :param inputs:
    :param N:
    :return:
    """
    usenum = inputs[0]  # 有效值
    i = 0  # 标记计数器
    for index, tmp in enumerate(inputs):
        if tmp != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs


def AmplitudeLimitingShakeOff(inputs, Amplitude, N):
    """
    限幅消抖滤波法
    Amplitude:	限制最大振幅
    N:			消抖上限
    :param inputs:
    :param Amplitude:
    :param N:
    :return:
    """
    # print(inputs)
    tmpnum = inputs[0]
    for index, newtmp in enumerate(inputs):
        if np.abs(tmpnum - newtmp) > Amplitude:
            inputs[index] = tmpnum
        tmpnum = newtmp
    # print(inputs)
    usenum = inputs[0]
    i = 0
    for index2, tmp2 in enumerate(inputs):
        if tmp2 != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index2] = usenum
    # print(inputs)
    return inputs


T = np.arange(0, 0.5, 1 / 4410.0)
num = signal.chirp(T, f0=10, t1=0.5, f1=1000.0)
plt.subplot(2, 1, 1)
plt.plot(num)

# result = ArithmeticAverage(num.copy(), 30)
result = MedianAverage(num.copy(), 30)
plt.subplot(2, 1, 2)
plt.plot(result)

plt.show()
