#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : kmeans++.py
@Time    : 2019/12/23 15:09
@desc	 : numpy原生实现K-Means++
'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    '''
    加载测试数据集，返回一个列表，列表的元素是一个坐标
    '''
    dataList = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataList.append(fltLine)
    return dataList


def euler_distance(point1: list, point2: list) -> float:
    """
    计算两点之间的欧拉距离，支持多维
    """
    distance = 0.0
    for a, b in zip(point1, point2):
        distance += math.pow(a - b, 2)
    return math.sqrt(distance)


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    for i, centroid in enumerate(centroids):
        dist = euler_distance(centroid, point)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def kpp_centers(data_set, k):
    """
    1.从数据集中返回 k 个对象可作为质心
    """

    # 将矩阵转为列表
    data_set = np.matrix.tolist(data_set)

    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers)  # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d):  # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break

    cluster_centers = np.mat(cluster_centers)
    return cluster_centers


def kMeans(dataSet, k):
    '''
    2.KMeans算法，返回最终的质心坐标和每个点所在的簇
    '''
    m = np.shape(dataSet)[0]  # m表示数据集的长度（个数）
    clusterAssment = np.mat(np.zeros((m, 2)))

    centroids = kpp_centers(dataSet, k)  # 保存k个初始质心的坐标
    clusterChanged = True
    iterIndex = 1  # 迭代次数
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = np.linalg.norm(np.array(centroids[j, :]) - np.array(dataSet[i, :]))
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
            print("第%d次迭代后%d个质心的坐标:\n%s" % (iterIndex, k, centroids))  # 第一次迭代的质心坐标就是初始的质心坐标
            iterIndex += 1
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    iter = (iterIndex - 1) / dataSet.shape[0]
    return centroids, clusterAssment, iter


def showCluster(dataSet, k, centroids, clusterAssment, iters):
    '''
    数据可视化,只能画二维的图（若是三维的坐标图则直接返回1）
    '''
    numSamples, dim = dataSet.shape
    if dim != 2:
        return 1

    mark = ['or', 'ob', 'og', 'ok', 'oy', 'om', 'oc', '^r', '+r', 'sr', 'dr', '<r', 'pr']

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Pr', 'Pb', 'Pg', 'Pk', 'Py', 'Pm', 'Pc', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.title('Number of Iterations: %d' % (iters))
    plt.show()


if __name__ == '__main__':
    # 加载数据集
    dataMat = np.mat(loadDataSet('./testSet'))  # mat是numpy中的函数，将列表转化成矩阵

    k = 4  # 选定k值，也就是簇的个数（可以指定为其他数）
    cent, clust, iters = kMeans(dataMat, k)

    # 数据可视化处理
    showCluster(dataMat, k, cent, clust, iters)
