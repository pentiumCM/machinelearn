#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : wheel_filter.py
@Time    : 2020/9/9 17:09
@desc	 : 
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def get_data(csv_path):
    """
    读取数据
    :param csv_path: 数据源的路径
    :return:
    """
    wheel_df = pd.read_csv(csv_path)

    sns.set()
    # Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
    sns.scatterplot(x=wheel_df['frame_index'], y=wheel_df['left_wheel_center_x'],
                    data=wheel_df, s=100)
    sns.relplot(x="frame_index", y="left_wheel_center_x", kind="line", data=wheel_df)
    plt.show()
    return wheel_df


def kalman_filter(measure_data):
    """
    卡尔曼滤波处理
    :param measure_data: 滤波数据（测量数据）
    :return:
    """
    # 构造具有所需尺寸的 filter
    kf = KalmanFilter(dim_x=2, dim_z=1)  # dim_x:隐状态大小，dim_z:量测大小

    # 为状态分配初始值
    # x = np.array([measure_data[0][0], measure_data[1][0]])
    x = np.array([0, 500])
    kf.x = x

    # 定义状态转换矩阵
    dt = 1
    F = np.array([[1, dt],
                  [0, 1]])
    kf.F = F

    # 定义状态协方差矩阵
    P = np.array([[1000, 20],
                  [20, 1000]])
    kf.P = P

    # 定义过程噪声
    kf.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

    # 定义测量矩阵
    kf.H = np.array([[1, 0]])

    # 定义测量噪声
    kf.R = 2

    iteration = measure_data.shape[1]

    filter_result = list()
    filter_result.append(x)
    for i in range(1, iteration):
        z = np.array(measure_data[1][i])
        kf.predict()
        kf.update(z)
        filter_result.append(kf.x)
    filter_result = np.squeeze(np.array(filter_result))

    return filter_result


def plot_result(measure_data, filter_result):
    plt.figure(figsize=(8, 4))
    plt.plot(range(0, measure_data.shape[1]), measure_data[1][0:], label='measure_data')
    plt.plot(range(0, filter_result.shape[0]), filter_result[0:, 0], label='Kalman Filter')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity [m]', fontsize=14)
    plt.show()


if __name__ == '__main__':
    df = get_data('F:/project/tx/three_wheel_test.csv')

    measure_data = np.array([df['frame_index'].values, df['left_wheel_center_x'].values])

    filter_result = kalman_filter(measure_data)

    plot_result(measure_data, filter_result)

    print(measure_data[1])
    print(filter_result[0:, 0])
