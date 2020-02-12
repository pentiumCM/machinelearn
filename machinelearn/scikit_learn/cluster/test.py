#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : test.py
@Time    : 2019/12/24 10:50
@desc	 :
'''

import numpy as np
import math

import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 1000)  # 这个表示在0到10之间生成1000个x值

# y=10*sin(5*x)+7*abs(x-5)+10
y = [i * np.sin(10 * math.pi * i) + 2 for i in x]  # 对上述生成的1000个数循环用函数公式求对应的y

plt.plot(x, y)  # 用上述生成的1000个xy值对生成1000个点

plt.show()  # 绘制图像

plt.savefig("chenminjpg")
