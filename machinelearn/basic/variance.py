#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : variance.py
@Time    : 2020/4/11 9:40
@desc	 : 均值，方差，标准差，协方差，相关系数
'''

import numpy as np

a = [1, 5, 6]
b = [4, 3, 9]
X = np.array([[1, 5, 6], [4, 3, 9], [4, 2, 9], [4, 7, 2]])

# 计算变量的平均数
aver_a = np.mean(a)
print("aver_a：", aver_a)

# 计算总体方差（有偏）
var_a = np.var(a)
print("var_a：", var_a)

# 计算总体标准差
std_a = np.std(a)
print("std_a：", std_a)

# 计算样本方差（无偏）
var_sample_a = np.var(a, ddof=1)
print("var_sample_a：", var_sample_a)

# 计算样本标准差
std_sample_a = np.std(a, ddof=1)
print("std_sample_a：", std_sample_a)

# 计算变量 a，b的协方差，（结果以协方差矩阵的形式呈现）
cov_ab = np.cov(a, b)
print("cov_ab：", cov_ab)

# 变量数组 X 的协方差矩阵
cov_X = np.cov(X)
print("cov_X：", cov_X)

# 计算变量a和b的相关系数
corrcoef_ab = np.corrcoef(a, b)
print("corrcoef_ab：", corrcoef_ab)


# 变量数组 X 的相关系数矩阵
corrcoef_X = np.corrcoef(X)
print("corrcoef_X：", corrcoef_X)