#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : lp_classify.py
@Time    : 2020/10/23 10:59
@desc	 : 构建车牌的分类器
'''

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

import joblib


def train_classification(dataset_path, usercols, model, model_type):
    """
    构建车牌分类器
    :param dataset_path:    数据集路径，针对于csv文件
    :param usercols:        特征列
    :param model:           选用的预训练的模型
    :param model_type:      模型类别，进行模型文件保存时用
    :return:
    """
    # 1. 加载数据
    filePath = dataset_path
    df = pd.read_csv(filepath_or_buffer=filePath, usecols=usercols)

    # 正负样本均衡操作
    pos_df = df[df["label"].isin([1])]
    neg_df = df[df["label"].isin([0])]
    ratio_pos2neg = pos_df.shape[0] / neg_df.shape[0]
    neg_df = neg_df.sample(frac=ratio_pos2neg)
    samples_df = pd.concat([pos_df, neg_df], axis=0)

    # 2. 数据标准化处理
    dataset = np.array(samples_df)

    # 标签一般在最后一列，所以找到最后一列的索引进行划分 x,y
    columns_end_index = len(usercols) - 1
    x = dataset[0:, 0:columns_end_index]
    y = dataset[0:, columns_end_index]
    scalar = StandardScaler().fit(x)
    x = scalar.transform(x)

    samples = y.shape[0]
    pos = np.sum(y)

    neg = samples - pos

    # 3. 构建数据集，训练集-测试集：7/3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # 4. 模型训练
    model.fit(x_train, y_train)

    # 5. 模型预测
    y_test_pred = model.predict(x_test)
    print("预测值：", y_test_pred)

    # 6. 模型评估
    score = accuracy_score(y_test, y_test_pred)
    print("准确率：", score)

    scores_cv = cross_val_score(model, x, y, cv=5)
    print("交叉验证法准确率：", scores_cv)
    print("交叉验证法准确率均值：", scores_cv.mean())

    # 7. 模型保存
    # 保存 model
    joblib.dump(model, model_type + '.pkl')

    # joblib.dump(scalar, 'scalar_f5.pkl')


if __name__ == "__main__":
    # 指定数据集和待训练的特征列
    dataset_path = 'F:/develop_code/python/machinelearn/machinelearn/scikit_learn/classification/dataset/sample_x_y_w_h_gray.csv'
    usercols = [1, 2, 3, 4, 5, 6]

    # 选取预训练的模型
    # clf = svm.SVC()
    clf = GaussianNB()
    # clf = AdaBoostClassifier(n_estimators=100)

    clf_type = 'gnb_f5'

    train_classification(dataset_path, usercols, clf, clf_type)
