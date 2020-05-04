#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : pca_demo.py
@Time    : 2020/4/14 11:22
@desc	 : PCA主成分分析demo
'''
import numpy as np
from sklearn.decomposition import PCA

f1 = [1, 2, 3, 4, 5]
f2 = [4, 5, 4, 5, 2]

data = np.array([f1, f2])

cov_f = np.cov(data)
print("协方差矩阵：", cov_f)
eigenvalue, featurevector = np.linalg.eig(cov_f)

print("特征值：", eigenvalue)
print("特征向量：", featurevector)

data = data.T
pca = PCA(n_components=1)
pca.fit(data)
print("降维后的数据：", pca.transform(data))

# 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
print("方差贡献率：", pca.explained_variance_ratio_)

# 降维后的各主成分的方差值
print("主成分方差：", pca.explained_variance_)
