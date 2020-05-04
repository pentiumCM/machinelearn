#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : iris_hierarchical_cluster.py
@Time    : 2020/4/15 23:55
@desc	 : 鸢尾花层次聚类
'''

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
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

from sklearn.decomposition import PCA

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

# 3. 绘制树状图
plt.figure()
Z = linkage(data_M, method='ward', metric='euclidean')
p = dendrogram(Z, 0)
plt.show()

# 4. 模型训练
ac = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward', affinity='euclidean')
ac.fit(data_M)

# 聚类
label_list = ac.fit_predict(data_M)
# 输出结果列表
format_print(label_list, 50)

print()

# 平面聚类的每一簇的元素
reslist = [[] for i in range(cluster_num)]
# 遍历聚类中每个簇的元素
for i in range(len(label_list)):
    label = label_list[i]
    # 遍历每一类
    reslist[label].append(data_M[i, :])

# 压缩点处理
# result = np.array(reslist)
# pca_center = PCA(n_components=1)
# for i in range(cluster_num):
#     cluster_i = np.array(result[i])
#     cluster_i = cluster_i.T
#     pca_center.fit(cluster_i)
#     cluster_i = pca_center.transform(cluster_i)
#
#     cluster_i = cluster_i.T
#     data_M = np.append(data_M, cluster_i)
# 压缩点结束

data_M = np.array(data_M.reshape((-1, 4)))
# 聚类结果可视化
pca_data = pac_transfer_data(data_M, 2)

# 绘制散点图
xd = pca_data[:, 0]
yd = pca_data[:, 1]

color_list = gener_colors(label_list)
for i in range(len(label_list), data_M.shape[0]):
    color_list.append('black')
plt_scatter2D(xd, yd, color_list, '层次聚类')

# 检验聚类的性能
# si系数在【-1,,1】之间，越大越好
cluster_score_si = metrics.silhouette_score(data_M, label_list)
print("cluster_score_si", cluster_score_si)

# ch分数越大越好
cluster_score_ch = metrics.calinski_harabasz_score(data_M, label_list)
print("cluster_score_ch：", cluster_score_ch)

# DBI的值最小是0，值越小，代表聚类效果越好。
cluster_score_DBI = metrics.davies_bouldin_score(data_M, label_list)
print("cluster_score_DBI ：", cluster_score_DBI)

# 划分聚类
print("k-means聚类结果：")
k_means = KMeans(n_clusters=cluster_num)
label_list_km = k_means.fit_predict(data_M)
print("label_list_km：")
format_print(label_list_km, 50)
color_list2 = gener_colors(label_list_km)
plt_scatter2D(xd, yd, color_list2, 'k-means')

cluster_score_si_km = metrics.silhouette_score(data_M, label_list_km)
print("cluster_score_si_km", cluster_score_si_km)

# ch分数越大越好
cluster_score_ch_km = metrics.calinski_harabasz_score(data_M, label_list_km)
print("cluster_score_ch_km：", cluster_score_ch_km)

# DBI的值最小是0，值越小，代表聚类效果越好。
cluster_score_DBI_km = metrics.davies_bouldin_score(data_M, label_list_km)
print("cluster_score_DBI_km ：", cluster_score_DBI_km)

# 密度聚类
labels_db = DBSCAN(eps=1, min_samples=8).fit_predict(data_M)
print("label_list_db：")
format_print(labels_db, 50)
color_list3 = gener_colors(labels_db)
plt_scatter2D(xd, yd, labels_db, '密度聚类')

cluster_score_si_db = metrics.silhouette_score(data_M, labels_db)
print("cluster_score_si_db", cluster_score_si_db)

# ch分数越大越好
cluster_score_ch_db = metrics.calinski_harabasz_score(data_M, labels_db)
print("cluster_score_ch_db：", cluster_score_ch_db)

# DBI的值最小是0，值越小，代表聚类效果越好。
cluster_score_DBI_db = metrics.davies_bouldin_score(data_M, labels_db)
print("cluster_score_DBI_db ：", cluster_score_DBI_db)
