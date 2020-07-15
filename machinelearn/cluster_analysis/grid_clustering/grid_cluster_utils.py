#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : grid_cluster_utils.py
@Time    : 2020/5/15 17:54
@desc	 : 网格聚类的工具类
'''

import numpy as np

from sklearn import metrics

from pyclustering.cluster.clique import clique
from pyclustering.cluster.clique import clique_visualizer

import time


def clique_excute(intervals, threshold, dataset):
    """
    使用网格聚类算法clique进行聚类
    :param intervals: 每维划分的网格数
    :param threshold: 密度阈值
    :param dataset: 聚类的数据集
    :return: 返回聚类的性能指标 - 轮廓系数
    """

    start = time.clock()  # 开始计时

    # 创建CLIQUE算法进行处理
    # 定义每个维度中网格单元的数量
    intervals = intervals
    # 密度阈值
    threshold = threshold
    clique_instance = clique(dataset, intervals, threshold)

    # 开始聚类过程并获得结果
    clique_instance.process()
    clique_cluster = clique_instance.get_clusters()  # allocated clusters
    noise = clique_instance.get_noise()  # points that are considered as outliers (in this example should be empty)
    cells = clique_instance.get_cells()  # CLIQUE blocks that forms grid
    print("Amount of clusters:", len(clique_cluster))
    print(clique_cluster)

    res = np.zeros(dataset.shape[0])
    for i in range(len(clique_cluster)):
        part = clique_cluster[i]
        for j in range(len(part)):
            res[part[j]] = i

    # 可视化聚类结果
    clique_visualizer.show_grid(cells, dataset)  # show grid that has been formed by the algorithm
    clique_visualizer.show_clusters(dataset, clique_cluster, noise)  # show clustering results

    # 轮廓系数的取值在【-1,1】之间，值越大，则聚类效果越好
    cluster_score_si_db = metrics.silhouette_score(dataset, res)
    print("cluster_score_si_db", cluster_score_si_db)

    runtime = (time.clock() - start)  # 结束计时
    print("running time:" + str(runtime) + "s")  # 印出时间

    return cluster_score_si_db, runtime
