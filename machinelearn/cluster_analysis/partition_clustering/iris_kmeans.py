#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : iris_hierarchical_cluster.py
@Time    : 2020/4/15 23:55
@desc	 : 鸢尾花划分聚类
'''

from sklearn import datasets
import numpy as np
from sklearn import preprocessing

from utils.pca_utils import pac_transfer_data
from utils.plt_utils import plt_scatter2D

# 划分聚类
from sklearn.cluster import KMeans
# 层次聚类
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
# 密度聚类
from sklearn.cluster import DBSCAN

from sklearn import metrics


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
std_scaler = preprocessing.StandardScaler()
data_M = std_scaler.fit_transform(data)

pca_data = pac_transfer_data(data_M, 2)

data_M = pca_data

# 划分聚类
print("k-means聚类结果：")
k_means = KMeans(n_clusters=cluster_num)
label_list_km = k_means.fit_predict(data_M)
print("label_list_km：")
format_print(label_list_km, 50)
color_list2 = gener_colors(label_list_km)
plt_scatter2D(pca_data[:, 0], pca_data[:, 1], 'k-means', color_list2)

# si系数【-1，1】，越大越好
cluster_score_si_km = metrics.silhouette_score(data_M, label_list_km)
print("cluster_score_si_km：", cluster_score_si_km)

# ch分数越大越好
cluster_score_ch_km = metrics.calinski_harabasz_score(data_M, label_list_km)
print("cluster_score_ch_km：", cluster_score_ch_km)

# DBI的值最小是0，值越小，代表聚类效果越好。
cluster_score_DBI_km = metrics.davies_bouldin_score(data_M, label_list_km)
print("cluster_score_DBI_km ：", cluster_score_DBI_km)
