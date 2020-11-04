#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : gaussian_nb.py
@Time    : 2020/10/20 0:04
@desc	 : 利用sklearn的高斯贝叶斯构建分类器
'''

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import joblib

features = ['x', 'y', 'width', 'height', 'B', 'G', 'R', 'label']

# 1. 加载数据
filePath_v1 = 'F:/develop_code/python/machinelearn/machinelearn/bayes/sample_v1.csv'
df1 = pd.read_csv(filepath_or_buffer=filePath_v1, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
print("df1:", df1.head())

filePath_v2 = 'F:/develop_code/python/machinelearn/machinelearn/bayes/sample.csv'
df2 = pd.read_csv(filepath_or_buffer=filePath_v2, usecols=[1, 2, 3, 4, 5, 6, 7, 8])
print("df2:", df2.head())

df = pd.concat([df1, df2], axis=0)
print("df:", df.head())

# 2. 数据预处理
# 2.2 数据去空处理
print("有无缺失值：")
print(df.isnull().any())
df.dropna(inplace=True)
dataset = np.array(df)
x = dataset[0:, 0:7]
y = dataset[0:, 7]

# 2.1 数据标准化处理
scalar = StandardScaler().fit(x)
x = scalar.transform(x)

# 3. 构建数据集，训练集-测试集：7/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 4. 模型训练
clf = svm.SVC(gamma='scale')
clf.fit(x_train, y_train)
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

# 5. 模型预测
y_test_pred = clf.predict(x_test)
print("预测值：", y_test_pred)

# 6. 模型评估
score = accuracy_score(y_test, y_test_pred)
print("准确率：", score)

scores_cv = cross_val_score(clf, x, y, cv=5)
print("交叉验证法准确率：", scores_cv)
print("交叉验证法准确率均值：", scores_cv.mean())

# 7. 模型保存
# 保存 model
joblib.dump(clf, 'svm.pkl')

# 加载 model
# clf = joblib.load('../../model/regr.pkl')
