#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : filter_tset.py
@Time    : 2020/9/8 20:55
@desc	 : python 内置的 filter 过滤函数
'''

import numpy as np

# filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
# 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。

# def is_odd(n):
#     return n % 2 == 1
#
#
# tmplist = filter(is_odd, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# newlist = list(tmplist)
# print(newlist)


f = np.array([[1, 0.1],
              [0, 1]])
x = np.array([10, 5])
res = np.dot(f, x)

print(res)
