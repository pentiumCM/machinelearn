#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : sklearn_test.py
@Time    : 2019/12/12 21:29
@desc	 :
'''

# 调用sklearn的数据集
from sklearn import datasets

# 调用SVM的支持向量分类
from sklearn import svm

digits = datasets.load_digits()

# print(digits.data)
# print(digits.target)

# 把估计器实例命名为 clf
clf = svm.SVC(gamma=0.001, C=100.)

# 估计器需要拟合模型, 通过将我们的训练集传递给 fit 方法来完成的。
clf.fit(digits.data[:-1], digits.target[:-1])

svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

print("训练完成，开始预测")

# 可以预测新的值，特别是我们可以向分类器询问 digits 数据集中最后一个图像
y = clf.predict(digits.data[-1:])

print("预测的结果胃为第几类：",y)
