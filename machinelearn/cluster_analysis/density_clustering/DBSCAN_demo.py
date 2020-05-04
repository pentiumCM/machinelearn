#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : DBSCAN_demo.py
@Time    : 2020/4/27 21:04
@desc	 : 密度聚类 - DBSCAN算法
'''

import numpy as np
from sklearn import datasets

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from sklearn import metrics
from utils.plt_utils import plt_scatter2D

# 生成一组随机数据
X1, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)

X = np.concatenate((X1, X2))
plt_scatter2D(x=X[:, 0], y=X[:, 1], title="原始数据")

cluster_num = 3

# 1. kmeans聚类
y_pred_km = KMeans(n_clusters=cluster_num, random_state=9).fit_predict(X)
plt_scatter2D(x=X[:, 0], y=X[:, 1], title="kmeans", colors=y_pred_km)

# 2. 层次聚类
ac = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward', affinity='euclidean')
ac.fit(X)
y_pred_ac = ac.fit_predict(X)
plt_scatter2D(x=X[:, 0], y=X[:, 1], colors=y_pred_ac, title="层次聚类")

# 3. 密度聚类
y_pred_db = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
plt_scatter2D(x=X[:, 0], y=X[:, 1], colors=y_pred_db, title="密度聚类")

# 性能检验
cluster_score_ch_km = metrics.calinski_harabasz_score(X, y_pred_km)
print("cluster_score_ch_km：", cluster_score_ch_km)

cluster_score_ch_ac = metrics.calinski_harabasz_score(X, y_pred_ac)
print("cluster_score_ch_ac：", cluster_score_ch_ac)

cluster_score_ch_db = metrics.calinski_harabasz_score(X, y_pred_db)
print("cluster_score_ch_db：", cluster_score_ch_db)
