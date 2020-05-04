#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : plt_learn.py
@Time    : 2020/4/24 16:00
@desc	 : matplotlib测试
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义图像和三维格式坐标轴
fig = plt.figure()
ax2 = Axes3D(fig)

z = np.linspace(0, 13, 1000)
x = 5 * np.sin(z)
y = 5 * np.cos(z)
zd = 13 * np.random.random(100)
xd = 5 * np.sin(zd)
yd = 5 * np.cos(zd)

# 绘制散点图
ax2.scatter3D(xd, yd, zd, cmap='Blues', c='g')
# 绘制空间曲线
ax2.plot3D(x, y, z, 'gray')
plt.show()
