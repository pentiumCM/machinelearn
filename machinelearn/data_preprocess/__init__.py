#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/4/22 22:29
@desc	 : 数据预处理模块
'''

list1 = [1, 2, 3, 4, 5]

length = len(list1)

# 是否已全部访问完
visited = True

j = 0
while (visited):
    for i in range(length):
        print('i = ', i)

        if (i == length - 1):
            print('长度一致')
            i = 0
            j += 1
        if (j == 2):
            visited = False
