#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : CLIQUE_demo.py
@Time    : 2020/4/28 22:26
@desc	 : CLIQUE 算法demo，由python的pyclustering模块提供
           参考：https://codedocs.xyz/annoviko/pyclustering/classpyclustering_1_1cluster_1_1clique_1_1clique.html
'''

import numpy as np
from sklearn import datasets

from utils.pca_utils import pac_transfer_data
from utils.scaler_utils import normalized_scaler_transdata

from pyclustering.cluster.clique import clique

# 可视化
from pyclustering.cluster.clique import clique_visualizer

from utils.plt_utils import plt_scatter2D

from sklearn.cluster import KMeans

# 定义常量
cluster_num = 3

# 1. 导入数据集
iris = datasets.load_iris()
iris_data = iris.data

# 构建训练数据
# f1 = np.array([37, 42, 49, 56, 61, 65])  # 体重
# f2 = np.array([147, 154, 161, 165, 172, 177])  # 身高
# f3 = np.array([9, 14, 20, 24, 30, 38])  # 年龄
# f4 = np.array([0, 0, 50, 100, 200, 300])  # 收入
# f5 = np.array([15, 15, 20, 25, 30, 50])  # 房间面积

f1 = np.array([0, 1.1, 1.7, 1.2, 2.5, 4])  # 体重
f2 = np.array([0, 1.1, 1.5, 2.5, 1.6, 4])  # 身高

data = np.array([f1, f2])
data = data.T

# 2. 数据预处理
# data = np.array(iris_data[:, 0:3])
# 2.1 归一化
# data_M = normalized_scaler_transdata(data)
data_M = data

pca_data = pac_transfer_data(data_M, n_components=2)

# 创建CLIQUE算法进行处理
# 定义每个维度中网格单元的数量
intervals = 4
# 密度阈值
threshold = 1
clique_instance = clique(data_M, intervals, threshold)

# 开始聚类过程并获得结果
clique_instance.process()
clique_cluster = clique_instance.get_clusters()  # allocated clusters
noise = clique_instance.get_noise()  # points that are considered as outliers (in this example should be empty)
cells = clique_instance.get_cells()  # CLIQUE blocks that forms grid
print("Amount of clusters:", len(clique_cluster))
print(clique_cluster)

# 可视化聚类结果
clique_visualizer.show_grid(cells, data_M)  # show grid that has been formed by the algorithm
clique_visualizer.show_clusters(pca_data, clique_cluster, noise)  # show clustering results

print("聚类结束")

y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(data_M)
print("y_pred：", y_pred)
plt_scatter2D(data_M[:, 0], data_M[:, 1], colors=y_pred, title='Kmeans')
