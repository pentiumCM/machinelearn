#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : syntax_test.py
@Time    : 2019/10/30 21:29
@desc	 : python语法测试文件
'''

import numpy as np

# 1. 激活函数
def sigmoid(z):
    """
    sigmoid激活函数
    :param z: 输入
    :return: sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    X = [0, 1.5, 2, 2, 2.25, 2.8, 3.2, 4]
    Y = [1.5, 3, 4, 3, 0.5, 2.8, 1.35, 4]

    X = np.array(X)
    Y = np.array(Y)

    data_X = np.vstack((X, Y))


    print("矩阵维度")
    X_shape = np.array(X)
    print(data_X)

    w = np.zeros((10, 1))
    print("矩阵：",w)
