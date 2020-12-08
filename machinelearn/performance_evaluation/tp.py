#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : tp.py
@Time    : 2020/11/12 10:06
@desc	 : 混淆矩阵测试
'''

# 导入依赖包
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

sns.set()
y_true = ["cat", "dog", "cat", "cat", "dog"]
y_pred = ["dog", "dog", "cat", "dog", "cat"]
C2 = confusion_matrix(y_true, y_pred, labels=["dog", "cat"])
sns.heatmap(C2, annot=True)

plt.show()
