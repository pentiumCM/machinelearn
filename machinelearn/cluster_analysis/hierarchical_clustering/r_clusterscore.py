#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : r_clusterscore.py
@Time    : 2020/4/16 17:27
@desc	 : 测试R的clusterscore函数
'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import csv

# 1. 导入数据集
datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\Z.csv'
testdisease_data = pd.read_csv(datafile)

header = testdisease_data.columns

# 2. 数据预处理
data = np.array(testdisease_data.values[0:10, 6:7])

std_data = data[0:, 0:1]
std = np.std(std_data)

print(std)

pca_data = np.array(testdisease_data.values[0:10, 1:6])

pca = PCA(n_components=1)
pca.fit(pca_data)
print("降维后的数据：")
pca_res = pca.transform(pca_data)
print(pca_res)

print("贡献率：",pca.explained_variance_ratio_)
