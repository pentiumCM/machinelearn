#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : test_result.py
@Time    : 2021/1/10 0:02
@desc	 :
'''

import os, sys

print(__file__)

# 获得绝对路径
print(os.path.abspath(__file__))

#
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 获得的是Test的绝对路径

print('')


