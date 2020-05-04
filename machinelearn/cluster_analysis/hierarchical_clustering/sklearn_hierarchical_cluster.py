#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : sklearn_hierarchical_cluster.py
@Time    : 2020/4/23 15:00
@desc	 : sklearn的层次聚类
'''

import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as dist
from scipy.cluster.hierarchy import dendrogram, linkage

data = np.array([[1, 2], [2, 3], [-3, 3], [-2, -1], [5, -1]])

# 画点
plt.scatter(x=data[:, 0:1], y=data[:, 1:2], marker='.', color='red')
n = np.arange(data.shape[0])
for i, txt in enumerate(n):
    plt.annotate(txt, (data[i:i + 1, 0:1], data[i:i + 1, 1:2]))
plt.show()

# 聚类方式一
# 训练模型
ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
clustering = ac.fit(data)

print("每个数据所属的簇编号：", clustering.labels_)
print("每个簇的成员：", clustering.children_)

# 聚类的方式二
# 自定义距离矩阵
num = data.shape[0]
dist_matrix = np.mat(np.zeros((num, num)))
for i in range(num):
    for j in range(i, num):
        distence = euclidean_distances(data[i:i + 1], data[j:j + 1])
        dist_matrix[i:i + 1, j:j + 1] = distence
        dist_matrix[j:j + 1, i:i + 1] = dist_matrix[i:i + 1, j:j + 1]

# 基于自定义的聚类矩阵进行聚类
model = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average')
clustering2 = model.fit(dist_matrix)

print("自定义距离矩阵聚类方式：")
print("每个数据所属的簇编号：", clustering2.labels_)
print("每个簇的成员：", clustering2.children_)

# 调整距离矩阵的形状
dist_matrix = dist.squareform(dist_matrix)

# linkage方法用于计算两个聚类簇s和t之间的距离d(s,t)
# 层次聚类编码为一个linkage矩阵。
Z = linkage(dist_matrix, 'average')
print("聚类过程：", Z)

# 将层级聚类结果以树状图表示出来
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()
