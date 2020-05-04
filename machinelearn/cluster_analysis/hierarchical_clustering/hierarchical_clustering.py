#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : hierarchical_clustering.py
@Time    : 2020/4/15 13:50
@desc	 : 层次聚类的demo
'''

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X = np.array([[1, 2], [2, 3], [-3, 3], [-2, -1], [5, -1]])

X = X.T

# 训练模型
clustering = AgglomerativeClustering().fit(X)

# 绘制树状图
plt.figure()
# 将层级聚类结果以树状图表示出来
Z = linkage(X, method='ward', metric='euclidean')
p = dendrogram(Z, 0)
plt.show()

print("hello cluster")
