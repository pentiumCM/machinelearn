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

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import joblib

# 1. 加载数据
filePath = 'F:/experiment/19/PythonCode/sample_v4.csv'
df = pd.read_csv(filepath_or_buffer=filePath, usecols=[1, 2, 3, 4, 5])
print(df.head())

# 2. 数据标准化处理
dataset = np.array(df)
x = dataset[0:, 0:4]
y = dataset[0:, 4]
scalar = StandardScaler().fit(x)
x = scalar.transform(x)

# 3. 构建数据集，训练集-测试集：7/3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

sum = np.sum(y_test)

# 4. 模型训练
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# 5. 模型预测
y_test_pred = gnb.predict(x_test)
print("预测值：", y_test_pred)

# 6. 模型评估
score = accuracy_score(y_test, y_test_pred)
print("准确率：", score)

# 7. 模型保存
# 保存 model
joblib.dump(gnb, 'gnb_f5.pkl')
joblib.dump(scalar, 'scalar_f5.pkl')

# 加载 model
# clf = joblib.load('../../model/regr.pkl')
