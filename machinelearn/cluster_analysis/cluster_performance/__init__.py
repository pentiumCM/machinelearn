#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/5/15 14:12
@desc	 : 聚类性能评估
'''

import numpy as np

from utils.loaddata_utils import load_csvdata
from utils.scaler_utils import std_scaler_transdata

from sklearn import metrics

# 1. 导入数据集
datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\subForest.csv'
testdisease_data = load_csvdata(datafile)

# 2. 数据预处理
data = np.array(testdisease_data, dtype=np.float16)

# 行代表维数，因为下面有转置操作
data = data[0:20, :]

# 数据标准化
data_std = std_scaler_transdata(data)

data_std = data_std.T

y_pred = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

# 轮廓系数的取值在【-1,1】之间，值越大，则聚类效果越好
cluster_score_si_db = metrics.silhouette_score(data_std, y_pred)
print("cluster_score_si_db", cluster_score_si_db)

# ch分数越大越好
cluster_score_ch_db = metrics.calinski_harabasz_score(data_std, y_pred)
print("cluster_score_ch_db：", cluster_score_ch_db)

# DBI的值最小是0，值越小，代表聚类效果越好。
cluster_score_DBI_db = metrics.davies_bouldin_score(data_std, y_pred)
print("cluster_score_DBI_db ：", cluster_score_DBI_db)
