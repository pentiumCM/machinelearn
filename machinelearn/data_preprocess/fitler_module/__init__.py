#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/9/8 20:55
@desc	 : 滤波处理
'''

# -*- coding: utf-8 -*-
'''
Created on 2018年1月24日
@author: Jason.F
@summary: 有监督回归学习-多项式回归，通过多项式特征拟合非线性关系
'''
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

if __name__ == "__main__":
    start = time.clock()

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None,
                     sep='\s+')
    df.columns = ['CRIM', 'ZM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                  'MEDV']
    X = df[['LSTAT']].values
    y = df['MEDV'].values
    regr = LinearRegression()
    # 创建多项式特征
    quadratic = PolynomialFeatures(degree=2)  # 二项式
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # 线性拟合
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    regr = regr.fit(X, y)
    linear_r2 = r2_score(y, regr.predict(X))
    y_lin_fit = regr.predict(X_fit)
    # 二项式拟合
    regr = regr.fit(X_quad, y)
    quadratic_r2 = r2_score(y, regr.predict(X_quad))
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    # 三项式
    regr = regr.fit(X_cubic, y)
    cubic_r2 = r2_score(y, regr.predict(X_cubic))
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    # 可视化效果
    plt.scatter(X, y, label='training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, label='linear(d=1),$R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
    plt.plot(X_fit, y_quad_fit, label='quadratic(d=2),$R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
    plt.plot(X_fit, y_cubic_fit, label='cubic(d=3),$R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000\'s [LSTAT]')
    plt.legend(loc='upper right')
    plt.show()

    end = time.clock()
    print('finish all in %s' % str(end - start))
