#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2019/12/17 10:19
@desc	 : 
'''

import time

result = time.time()

print('result：', result)

year = result / (365 * 24 * 60 * 60) + 1970

print("year：", year)
