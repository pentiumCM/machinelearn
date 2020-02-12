#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : syntax_test.py
@Time    : 2019/10/30 21:29
@desc	 : python语法测试文件
'''

import numpy as np

x = np.array([3.6, 4.5, 2.6, 4.9, 2.5, 3.5])
print("x:",x)

y = np.reshape(x,(-1,1))
print("y:",y)