#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : test_reg.py
@Time    : 2020/9/10 20:26
@desc	 : 
'''

import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

from utils.loaddata_utils import load_csvdata

import joblib as jl


def polynomial_regression(data):
    """
    多项式回归
    :param data: 待拟合的数据
    :param degree: 多项式的次数
    :return:
    """
    data_x = np.reshape(data[0], (-1, 1))
    data_y = np.reshape(data[1], (-1, 1))

    # #degree=2表示建立datasets_X的二 次多项式特征X_poly。
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)

    # 创建多项式特征
    x_quad = quadratic.fit_transform(data_x)
    x_cubic = cubic.fit_transform(data_x)

    # 加载线性模型
    regr = jl.load('regr3.pkl')

    # 在线性模型中使用
    # 线性拟合
    # regr = regr.fit(data_x, data_y)
    # linear_r2 = r2_score(data_y, regr.predict(data_x))
    # y_line_fit = regr.predict(data_x)

    # 二项式拟合
    # regr = regr.fit(x_quad, data_y)
    # quadratic_r2 = r2_score(data_y, regr.predict(x_quad))
    # y_quad_fit = regr.predict(x_quad)

    # 三项式拟合
    regr = regr.fit(x_cubic, data_y)
    cubic_r2 = r2_score(data_y, regr.predict(x_cubic))
    y_cubic_fit = regr.predict(x_cubic)

    # 可视化效果
    plt.scatter(data_x, data_y, label='training points', color='gray')
    # plt.plot(data_x, y_line_fit, label='linear(d=1),$R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
    # plt.plot(data_x, y_quad_fit, label='quadratic(d=2),$R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
    plt.plot(data_x, y_cubic_fit, label='cubic(d=3),$R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
    plt.xlabel('frame_index')
    plt.ylabel('wheel_x')
    plt.legend(loc='upper right')
    plt.show()

    # return data_x, y_quad_fit


if __name__ == '__main__':
    df = load_csvdata('F:/project/tx/filter/two_wheel.csv')

    sns.set()
    # Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
    sns.scatterplot(x=df['frame_index'], y=df['wheel_centers'],
                    data=df, s=50)
    sns.relplot(x="frame_index", y="wheel_centers", kind="line", data=df)
    plt.show()

    data = np.array([df['frame_index'].values, df['wheel_centers'].values])
    polynomial_regression(data)

    print(df.tail())
