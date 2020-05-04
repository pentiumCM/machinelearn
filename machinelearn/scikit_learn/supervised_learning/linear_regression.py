#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : linear_regression.py
@Time    : 2019/12/19 0:19
@desc	 : 线性回归示例
'''

from matplotlib import pyplot as plot
import numpy as np
from sklearn import linear_model
import pickle

# 构建训练数据
X = np.array([48, 57, 50, 54, 64, 61, 43, 59])  # 体重
Y = np.array([165, 165, 157, 170, 175, 165, 155, 170])  # 身高

# 调整训练数据的维度
X_train = np.reshape(X, (-1, 1))
Y_train = np.reshape(Y, (-1, 1))

# 建立线性模型
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# 保存 model
f = open('F:/develop_code/python/machinelearn/docs/model/regr2.pkl', 'wb')
pickle.dump(regr, f)
f.close()

# 加载 model
f = open('F:/develop_code/python/machinelearn/docs/model/regr2.pkl', 'rb')
clf = pickle.load(f)
f.close()

Y_pred = clf.predict(X_train)

# 可视化
# 1.训练集数据
plot.scatter(X_train, Y_train, color='green')

# 2.测试数据
plot.plot(X_train, Y_pred, color='red')

plot.show()
