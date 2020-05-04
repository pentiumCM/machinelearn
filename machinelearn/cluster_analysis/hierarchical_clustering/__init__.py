#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/4/15 10:45
@desc	 : 层次聚类法
'''

# !/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : testDisease_cluster.py
@Time    : 2020/4/16 0:06
@desc	 : 测试病的层次聚类
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing

import csv

# 1. 导入数据集
datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\subForest.csv'
testdisease_data = pd.read_csv(datafile)
# print(testdisease_data)


header = testdisease_data.columns

# 2. 数据预处理
data = np.array(testdisease_data.values)

scaler = preprocessing.StandardScaler().fit(data)

X_scaled = scaler.transform(data)

# print(data)


with open('subForest_scaled.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(header)
    f_csv.writerows(X_scaled)
