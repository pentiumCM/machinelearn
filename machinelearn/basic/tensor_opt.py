#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : tensor_opt.py
@Time    : 2020/4/11 9:42
@desc	 : 张量运算
'''

import numpy as np

a = np.array([[1, 1], [1, 1]])

b = np.array([[1, 2], [3, 4]])

x = np.array([1,2])
y = np.array([3,4])

add = a + b

dot = np.dot(x, y)


print(add)
print(dot)