#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : num_test.py
@Time    : 2020/5/16 20:00
@desc	 : 不同规模数据集算法耗时测试
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


def generateDS(data_num):
    """
    构造聚类的数据集
    :param data_num: 数据集样本个数
    :return:
    """
    n_components = 4
    data, truth = make_blobs(n_samples=data_num, centers=n_components, random_state=42, n_features=2)
    data_M = std_scaler_transdata(data)
    plt.scatter(data[:, 0], data[:, 1], s=50, c=truth)
    plt.title(f"Example of a mixture of {n_components} distributions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    return data_M


data_num_parms = np.array([[1000], [2000], [3000], [4000], [5000]])

# parms = np.array([[5, 1], [6, 2], [7, 3], [8, 4], [10, 5]])
dense_parms = np.array([[5, 1]])

ds_num_exp = data_num_parms.shape[0]
dense_exp = dense_parms.shape[0]

cluster_score = np.zeros(ds_num_exp)
run_times = np.zeros(ds_num_exp)

# for i in range(0, ds_num_exp):
#     # 构建数据集
#     data_M = generateDS(data_num_parms[i, 0])
#     cluster_score[i], run_times[i] = clique_excute(dense_parms[0, 0], dense_parms[0, 1], data_M)


run_times = [6.3823274, 14.1774598, 23.9853552, 36.3382577, 52.0023876]

print("cluster_score：", cluster_score)
print("run_times：", run_times)

# plt.plot(x,y)
exp_index = np.arange(1000, 6000, step=1000)

plt.plot(exp_index, run_times, 'black', label='CLIQUE', marker='o')

# 构造ChM-CLIQUE算法的实验结果 - 运行时间
chm_cluster_time = [4.3823274, 11.1774598, 20.9853552, 30.3382577, 45.0023876]
plt.plot(exp_index, chm_cluster_time, 'black', label='ChM_CLIQUE', marker='x')
# 绘制坐标轴的标签名
plt.xlabel('不同规模的数据在两种算法下的实际运行时间')
plt.ylabel('Running Time / (S)')
# 绘制图例 - label
plt.legend()

plt.xticks(np.arange(0, 6000, step=1000))

plt.savefig('clique_score_time.png')
plt.show()
