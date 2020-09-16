#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : build_model.py
@Time    : 2019/12/24 10:50
@desc	 : 聚类分析 - K-means
            参考博客：https://blog.csdn.net/weixin_40683253/article/details/81288900
'''
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

# 一、读取文件
# 文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
# datafile = u'iris.csv'
datafile = u'F:\\develop_code\\python\\machinelearn\\docs\\dataset\\testDisease.csv'
# 设置输出文件的位置
outfile = u'cluster_out_disease.xlsx'
data = pd.read_csv(datafile)
d = DataFrame(data)
d.head()

print(d.head())

# 二、聚类
# 聚成3类数据,并发数为4，最大循环次数为500
mod = KMeans(n_clusters=3, n_jobs=4, max_iter=500)
mod.fit_predict(d)  # y_pred表示聚类的结果

# 聚成3类数据，统计每个聚类下的数据量，并且求出他们的中心
r1 = pd.Series(mod.labels_).value_counts()
r2 = pd.DataFrame(mod.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
r.columns = list(d.columns) + [u'类别数目']
print(r)

# 给每一条数据标注上被分为哪一类
r = pd.concat([d, pd.Series(mod.labels_, index=d.index)], axis=1)
r.columns = list(d.columns) + [u'聚类类别']
print(r.head())
# 如果需要保存到本地，就写上这一列
r.to_excel(outfile)

# 三、可视化过程
from sklearn.manifold import TSNE

ts = TSNE()
ts.fit_transform(r)
ts = pd.DataFrame(ts.embedding_, index=r.index)

import matplotlib.pyplot as plt

a = ts[r[u'聚类类别'] == 0]
plt.plot(a[0], a[1], 'r.')
a = ts[r[u'聚类类别'] == 1]
plt.plot(a[0], a[1], 'go')
a = ts[r[u'聚类类别'] == 2]
plt.plot(a[0], a[1], 'b*')
plt.show()
