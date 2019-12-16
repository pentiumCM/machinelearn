#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : GA_BP.py
@Time    : 2019/12/13 16:17
@desc	 : 基于遗传算法的BP模型
'''

# 调用python的数据集
from sklearn import datasets

# MLPClassifier 类实现了通过 Backpropagation 进行训练的多层感知器（MLP）算法
from sklearn.neural_network import MLPClassifier

datasets.load_iris();

X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

# MLP 在两个 array 上进行训练
clf.fit(X, y)


MLPClassifier(hidden_layer_sizes=(5, 2),activation='relu',
			  solver='lbfgs',alpha=1e-5,
			  batch_size='auto',learning_rate='constant',
			  learning_rate_init=0.001,power_t=0.5,max_iter=200,
			  shuffle=True,random_state=1,tol=1.e-4,
			  verbose=False, warm_start=False,momentum=0.9,
			  nesterovs_momentum=True, early_stopping=False,
			  validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
              epsilon=1e-8,n_iter_no_change=10)

labels = clf.predict([[2., 2.], [-1., -2.]])

print("预测结束",labels)