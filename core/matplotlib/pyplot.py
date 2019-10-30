#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : pyplot.py
@Time    : 2019/10/28 20:58
@desc	 : 
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10,10,500)
y = 1 / (1 + np.exp(-x))

plt.figure()
plt.plot(x,y)
plt.show()