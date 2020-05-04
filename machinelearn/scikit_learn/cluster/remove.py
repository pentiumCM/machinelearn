#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : remove.py
@Time    : 2020/4/13 20:26
@desc	 : 数据集去空处理
'''

import pandas as pd
import numpy as np

datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\testDisease.csv'
data_test = pd.read_csv(datafile)

list = np.isnan(data_test).any()

print(np.isnan(data_test).any())


if np.isnan(data_test).any() == 'true':
    print("有缺失值")