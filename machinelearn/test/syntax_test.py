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

list1 = [1, 2, 3,
         4, 5, 6,
         7, 8, 9]

segment_index = [0, 3, 6, 8]

# for item in range(1, len(segment_index)):
#     # 获取第 i 段的索引
#     segment_start_index = segment_index[item - 1]
#     segment_end_index = segment_index[item] - 1
#
#     if segment_index[item] == len(list1) - 1:
#         segment_end_index = segment_index[item]
#
#     print("当前为第几段：", item)
#     print("段元素{start},{end}：".format(start=list1[segment_start_index], end=list1[segment_end_index]))

test = list1[0:2]

print(test)
