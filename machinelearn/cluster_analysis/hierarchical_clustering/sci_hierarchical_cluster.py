#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : sci_hierarchical_cluster.py
@Time    : 2020/4/15 22:21
@desc	 : scipy实现层次聚类
'''

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from matplotlib import pyplot as plt

data = np.array([[1, 2], [2, 3], [-3, 3], [-2, -1], [5, -1]])

# 画点
plt.scatter(x=data[:, 0:1], y=data[:, 1:2], marker='.', color='red')
n = np.arange(data.shape[0])
for i, txt in enumerate(n):
    plt.annotate(txt, (data[i:i + 1, 0:1], data[i:i + 1, 1:2]))
plt.show()

# 1. 层次聚类
# linkage方法用于计算两个聚类簇s和t之间的距离d(s,t)
# 层次聚类编码为一个linkage矩阵。
Z = linkage(data, 'average')
print("聚类过程：", Z)

# 从给定链接矩阵定义的层次聚类中形成平面聚类。
f = fcluster(Z, 4, 'distance')
print("平面聚类结果：", f)


# 将层级聚类结果以树状图表示出来
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()
