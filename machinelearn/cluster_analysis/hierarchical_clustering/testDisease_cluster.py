#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : testDisease_cluster.py
@Time    : 2020/4/16 0:06
@desc	 : 测试病的层次聚类
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn import metrics

from utils.scaler_utils import std_scaler_transdata

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

# 1. 导入数据集
datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\subForest.csv'
testdisease_data = pd.read_csv(datafile)

# 读取四诊变量的映射文件
mapfile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\hyper.xlsx'
map_data = pd.read_excel(io=mapfile, sheet_name="信息导入", index_col=[2, 3])
map_data = np.array(map_data)

# 2. 数据预处理
data = np.array(testdisease_data)

# 数据标准化
data_M = std_scaler_transdata(data)

data_M = data_M.T

# 数据降维操作
# pca = PCA(n_components=82)
# pca.fit(data_M)
#
# data_M = pca.transform(data_M)
#
# print("贡献率：",pca.explained_variance_ratio_)

# 加一个标签进行区分
encode_list = ["S2011500", "S1021300", "S1040300", "S3310100", "S3302700", "S3322000", "S1070124", "S1070140",
               "S1070218",
               "SZ340100", "SZ110300", "SZ344100", "SZ344000", "SZ344200", "SZ110700", "SZ210800", "SZ311200",
               "SZ370100",
               "SZ370200", "SZ340200", "SZ310700", "SZ320500", "SZ320600", "SZ210900", "SZ343500", "SZ132900",
               "SZ344310",
               "SZ420300", "SZ330100", "SZ340400", "SZ340500", "SZ340600", "SZ340800", "SZ321200", "SZ310800",
               "SZ120800",
               "SZ340900", "SZ341000", "SZ141200", "SZ341300", "SZ341200", "SZ341500", "SZ341100", "SZ341700",
               "SZ341800",
               "SZ350600", "SZ350800", "SZ160430", "SZ342000", "SZ142600", "SZ141500", "SZ330700", "SZ344330",
               "SZ131900",
               "SZ332000", "SZ132100", "SZ343000", "SZ351200", "SZ351300", "SZ330960", "SZ331000", "SZ330900",
               "SZ361600",
               "SZ361200", "SZ361900", "SZ361500", "SZ360100", "SZ170112", "SZ170113", "SZ170114", "SZ170115",
               "SZ170122",
               "SZ170211", "SZ170215", "SZ170221", "SZ170222", "SZ410600", "SZ410400", "SZ411300", "SZ410900",
               "SZ170111",
               "S3311100"]

# 匹配编码名称
codename_list = []
map_data_num = map_data.shape[0]
for encode in encode_list:
    for i in range(0, map_data_num):
        map_encode = map_data[i:i + 1, 0: 1]
        if encode == map_encode:
            codename_list.append(map_data[i:i + 1, 1: 2])
            break

# 3. 绘制树状图
plt.figure(figsize=(30, 6))
Z = linkage(data_M, method='average', metric='euclidean')
p = dendrogram(Z, 0, labels=encode_list)
plt.savefig('D:\\test3.png')
plt.show()

# 4. 模型训练
cluster_num = 3
ac = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward', affinity='euclidean')
ac.fit(data_M)

# 聚类
label_list = ac.fit_predict(data_M)
print("平面聚类结果的簇编码：", label_list)

# # 聚类结果可视化
# plt.scatter(data_M[:, 0], data_M[:, 1], c=label_list)
# plt.show()

reslist = [[] for i in range(cluster_num)]
# 遍历聚类中每个簇的元素
for i in range(len(label_list)):
    label = label_list[i]
    # 遍历每一类
    reslist[label].append(codename_list[i])

# 输出平面聚类的结果
for i in range(cluster_num):
    print("第", i + 1, "簇的元素：")
    list = reslist[i]
    for j in range(len(list)):
        print(list[j], end=" ")
    print("\n")

# 检验聚类的性能
# metrics.silhouette_score(X, labels[, …])
# 轮廓系数取值在【-1,1】之间，值越大，聚类效果越好
python_cluster_score = metrics.silhouette_score(data_M, label_list)
print("python_cluster_score：", python_cluster_score)

# ch分数越大越好
cluster_score_ch = metrics.calinski_harabasz_score(data_M, label_list)
print("cluster_score_ch：", cluster_score_ch)

# DBI的值最小是0，值越小，代表聚类效果越好。
cluster_score_DBI = metrics.davies_bouldin_score(data_M, label_list)
print("cluster_score_DBI ：", cluster_score_DBI)

print()

R_labels = [1, 2, 2, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1,
            2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 3, 3, 1, 3, 3,
            3, 3, 3, 3, 3, 2, 2, 3, 3, 1]

R_cluster_score = metrics.silhouette_score(data_M, R_labels)

print("R_cluster_score：", R_cluster_score)
