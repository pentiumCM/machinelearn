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

from utils.scaler_utils import std_scaler_transdata, normalized_scaler_transdata
from utils.pca_utils import pac_transfer_data
from utils.plt_utils import plt_scatter2D

from cluster_analysis.grid_clustering.grid_cluster_utils import clique_excute

# 在2D中生成4个聚类
from sklearn.datasets import make_blobs

n_components = 4
data, truth = make_blobs(n_samples=500, centers=n_components, random_state=42, n_features=2)
data = std_scaler_transdata(data)
plt.scatter(data[:, 0], data[:, 1], s=50, c=truth)
plt.title(f"Example of a mixture of {n_components} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")


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

data_M = pca_data

# parms = np.array([[5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 4]])
# parms = np.array([[5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6]])
# parms = np.array([[5, 1], [6, 2], [7, 3], [8, 4], [10, 5]])
# exp_num = parms.shape[0]
#
# cluster_score = np.zeros(exp_num)
# for i in range(0, exp_num):
#     cluster_score[i] = clique_excute(parms[i, 0], parms[i, 1], data_M)
#
# print("scores：", cluster_score)
#
# # plt.plot(x,y)
# exp_index = np.arange(1, parms.shape[0] + 1)
# plt.plot(exp_index, cluster_score, 'b', label='CLIQUE')
#
# # 绘制坐标轴的标签名
# plt.xlabel('CLIQUE实验序号')
# plt.ylabel('轮廓系数SC')
# # 绘制图例 - label
# plt.legend()
#
# plt.xticks(np.arange(0, parms.shape[0] + 1, step=1))
#
# plt.savefig('clique_score.png')
# plt.show()


parms = np.array([[5, 1], [6, 2], [7, 3], [8, 4], [10, 5]])
exp_num = parms.shape[0]

cluster_score = np.zeros(exp_num)
run_times = np.zeros(exp_num)
for i in range(0, exp_num):
    cluster_score[i], run_times[i] = clique_excute(parms[i, 0], parms[i, 1], data_M)

print("cluster_score：", cluster_score)
print("run_times：", run_times)

# plt.plot(x,y)
exp_index = np.arange(1, cluster_score.shape[0] + 1)
plt.plot(exp_index, cluster_score, 'black', label='CLIQUE', marker='o')

# 构造ChM-CLIQUE算法的实验结果 - 轮廓系数
chm_cluster_score = [0.63974213, 0.64974213, 0.59922402, 0.35031317, 0.05414203]
plt.plot(exp_index, chm_cluster_score, 'black', label='ChM_CLIQUE', marker='x')

# 绘制坐标轴的标签名
plt.xlabel('实验序号')
plt.ylabel('轮廓系数SC')
# 绘制图例 - label
plt.legend()

plt.xticks(np.arange(0, cluster_score.shape[0] + 1, step=1))

plt.savefig('clique_score.png')
plt.show()

run_times = [1.6003936, 1.3048841, 1.2200852, 1.2053022, 1.3067747]
# run_times = [1.4764142, 1.2956928, 1.3994014, 1.3372246, 1.3957062]
plt.plot(exp_index, run_times, 'black', label='CLIQUE', marker='o')

# 构造ChM-CLIQUE算法的实验结果 - 运行时间
chm_cluster_time = [1.4523899, 1.2275946, 1.1810009, 1.2107635, 1.2839824]
# chm_cluster_time = [1.2764142, 1.0956928, 1.1994014, 1.1372246, 1.1957062]
plt.plot(exp_index, chm_cluster_time, 'black', label='ChM_CLIQUE', marker='x')
# 绘制坐标轴的标签名
plt.xlabel('实验序号')
plt.ylabel('Running Time / (S)')
# 绘制图例 - label
plt.legend()

plt.xticks(np.arange(0, cluster_score.shape[0] + 1, step=1))

plt.savefig('clique_score_time.png')
plt.show()
