#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : KalmanFilter_test.py
@Time    : 2020/9/8 21:40
@desc	 : KalmanFilter 测试
'''

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

np.random.seed(2)


def demo():
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    Q = 1e-2 * np.array([[1 / 4 * dt ** 4, 1 / 2 * dt ** 3], [1 / 2 * dt ** 3, dt ** 2]])
    R = 2.
    itr = 100

    # 实际结果
    real_state = []
    x = np.array([10, 5]).reshape(2, 1)

    for i in range(itr):
        real_state.append(x[0, 0])
        x = np.dot(F, x) + np.random.multivariate_normal(mean=(0, 0), cov=Q).reshape(2, 1)

    # 测量结果
    measurements = [x + np.random.normal(0, R) for x in real_state]

    # initialization
    P = np.array([[10, 5], [5, 10]])
    x = np.random.multivariate_normal(mean=(10, 5), cov=P).reshape(2, 1)

    # filter
    kf = KalmanFilter(dim_x=2, dim_z=1)  # dim_x:隐状态大小，dim_z:量测大小
    # 定义参数
    kf.x = x  # 初始状态[位置,速度]
    kf.F = F  # 状态转移矩阵
    kf.H = np.array([[1., 0.]])  # 量测矩阵
    kf.P = P  # 初始状态协方差
    kf.R = R  # 量测噪声
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=1e-2)  # 过程（系统）噪声

    filter_result = list()
    filter_result.append(x)
    for i in range(1, itr):
        z = measurements[i]
        kf.predict()
        kf.update(z)
        filter_result.append(kf.x)
    filter_result = np.squeeze(np.array(filter_result))

    return measurements, real_state, filter_result


def plot_result(measurements, real_state, filter_result):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(measurements)), measurements[1:], label='Measurements')
    plt.plot(range(1, len(real_state)), real_state[1:], label='Real statement')
    plt.plot(range(1, len(filter_result)), np.array(filter_result)[1:, 0], label='Kalman Filter')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.axhline(5, label='Real statement')  # , label='$GT_x(real)$'
    plt.plot(range(1, len(filter_result)), np.array(filter_result)[1:, 1], label='Kalman Filter')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)
    plt.show()


if __name__ == '__main__':
    measurements, real_state, filter_result = demo()
    plot_result(measurements, real_state, filter_result)
