#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : res_index.py
@Time    : 2020/5/24 16:54
@desc	 : 实验结果系数
'''

from sklearn.metrics.cluster import adjusted_rand_score

score = adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])

score1 = adjusted_rand_score([1, 1, 0, 0], [0, 0, 1, 1])
score2 = adjusted_rand_score([1, 1, 1, 1], [1, 1, 0, 0])  # ARI is symmetri c

print(score)
print(score1)
print(score2)
