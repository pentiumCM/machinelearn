#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : iris_hierarchical_cluster.py
@Time    : 2020/4/15 23:55
@desc	 : 鸢尾花网格聚类 - clique算法实现
'''

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from utils.scaler_utils import std_scaler_transdata
from utils.pca_utils import pac_transfer_data
from utils.plt_utils import plt_scatter2D

from pyclustering.cluster.clique import clique
from pyclustering.cluster.clique import clique_visualizer


def gener_colors(labels):
    """
    构建每个簇点的颜色
    :param labels:
    :return:
    """
    colors = []
    for label in labels:
        if label == 0:
            colors.append('r')
        elif label == 1:
            colors.append('y')
        elif label == 2:
            colors.append('g')
        elif label == 3:
            colors.append('violet')
    return colors


def format_print(labels, r_size):
    """
    格式化输出结果
    :param labels: 待输出结果的列表
    :param r_size: 每一行元素的个数
    :return:
    """
    for i in range(len(labels)):
        if i % r_size == 0:
            print()
        else:
            print(labels[i], end=" ")
    print()


# 定义常量
cluster_num = 3

# 1. 导入数据集
iris = datasets.load_iris()
iris_data = iris.data

# 2. 数据预处理
data = np.array(iris_data)
# 2.1 数据标准化
data_M = std_scaler_transdata(data)
# 2.2 数据降维
pca_data = pac_transfer_data(data_M, 2)

# 创建CLIQUE算法进行处理
# 定义每个维度中网格单元的数量
intervals = 10
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
