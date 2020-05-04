#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : plt_utils.py
@Time    : 2020/4/29 21:15
@desc	 : plt工具模块
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def plt_scatter2D(x, y, title, colors=[]):
    """
    绘制二维散点图
    :param x:
    :param y:
    :param colors:
    :param title:
    :return:
    """
    if len(colors):
        plt.scatter(x, y, c=colors)
    else:
        plt.scatter(x, y)
    plt.title(title)
    plt.show()


def plt_scatter3D(x, y, z, colors, title):
    """
    画三维散点图
    :param x:  x轴坐标
    :param y:  y轴坐标
    :param z:  z轴坐标
    :param colors: 每个点对应的颜色列表
    :param title:  图的标题
    :return:
    """
    # 定义三维坐标轴
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(x, y, z, cmap='Blues', c=colors)
    plt.title(title)
    plt.show()
