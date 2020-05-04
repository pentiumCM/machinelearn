#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/4/28 22:23
@desc	 : 基于网格的聚类
'''

# !/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : CLIQUE_demo.py
@Time    : 2020/4/28 22:26
@desc	 : CLIQUE 算法demo
           参考：https://codedocs.xyz/annoviko/pyclustering/classpyclustering_1_1cluster_1_1clique_1_1clique.html
'''

import os
import sys

import numpy as np
import pandas as pd
from functools import reduce

import seaborn as sns
from collections import Counter
import itertools

# 直方图
import scipy.sparse.csgraph
from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler

n_components = 4
data, truth = make_blobs(n_samples=100, centers=n_components, random_state=42, n_features=2)

data = MinMaxScaler().fit_transform(data)
plt.scatter(data[:, 0], data[:, 1], s=50, c=truth)
plt.title(f"Example of a mixture of {n_components} distributions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
