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
from sklearn.cluster import AgglomerativeClustering

from utils.scaler_utils import std_scaler_transdata
from utils.pca_utils import pac_transfer_data
from utils.loaddata_utils import load_csvdata, load_exceldata

# 网格聚类
from pyclustering.cluster.clique import clique
from pyclustering.cluster.clique import clique_visualizer

from sklearn import metrics

# 1. 导入数据集
datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\subForest.csv'
testdisease_data = load_csvdata(datafile)

# 读取四诊变量的映射文件
mapfile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\hyper.xlsx'

map_data = load_exceldata(mapfile, '信息导入', index_col=[2, 3])
map_data = np.array(map_data)

# 2. 数据预处理
data = np.array(testdisease_data)

data = data[0:10, 0:5]

# 数据标准化
data_M = std_scaler_transdata(data)

data_M = data_M.T

# 数据降维操作
pca_data = pac_transfer_data(data_M, 0.9)

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

# 创建CLIQUE算法进行处理
# 定义每个维度中网格单元的数量
intervals = 5
# 密度阈值
threshold = 0
clique_instance = clique(data_M, intervals, threshold)

# 开始聚类过程并获得结果
clique_instance.process()
clusters = clique_instance.get_clusters()  # allocated clusters
# 被视为离群值的点（噪点）
noise = clique_instance.get_noise()
# 形成网格的CLIQUE块
cells = clique_instance.get_cells()
print("Amount of clusters:", len(clusters))

# 可视化聚类结果
# 显示由算法形成的网格单元
clique_visualizer.show_grid(cells, pca_data)
# 显示聚类结果
clique_visualizer.show_clusters(pca_data, clusters, noise)

print("聚类结束")
