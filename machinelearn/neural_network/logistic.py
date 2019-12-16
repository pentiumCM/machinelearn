#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : logistic.py
@Time    : 2019/10/27 21:49
@desc	 : 基于逻辑回归的单层感知器(神经网络)算法
'''

import numpy as np
import matplotlib.pyplot as plt


# 1. 激活函数
def sigmoid(z):
    """
    sigmoid激活函数
    :param z: 输入
    :return: sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))


# 2.初始化参数,对于logistic回归,可以将权重初始化为零
def init_param_withZeros(dim):
    """
    初始化权重和偏置
    :param dim: 输入维度
    :return: 返回初始化的w和b
              w:(dim,1)的向量
              b:标量
    """
    w = np.zeros((dim, 1))
    b = 0
    return w, b


# 3.正向传播函数
def forward_propagate(w, b, X, Y):
    """
    BP算法分为两个部分：正向传播与反向传播。正向传播：计算出神经网络的输出。反向传播是采用梯度下降法使误差函数减小

    :param w: 权重向量
    :param b: 偏置
    :param X: 输入数据向量
    :param Y: 输入标签。逻辑回归是二分类问题,Y为0/1
    :return: 梯度和损失函数的值
    """
    # 输入数据数目
    num = X.shape[1]

    Z = np.dot(w.T, X) + b  # z = wTX + b.使用向量化同时计算，消除了代码中的显式的for循环
    # 正向传播
    A = sigmoid(Z)  # 整个训练集的预测值 A = [a1,a2,......,am]

    # 损失函数,损失函数采用交叉熵算法。
    loss = Y * np.log(A) + (1 - Y) * np.log(1 - A)
    # 成本函数是w和b的函数,是1到m项损失函数的平均,衡量了参数w,b在训练集上的效果
    cost = -1 / num * np.sum(loss)

    # 反向传播,求出权值w和偏置b的导数
    dz = A - Y  # dz = [a1-y1,.....,am-ym]
    dw = 1 / num * np.dot(X, dz.T)  # dw = 1/m * [x1 * dz1 + ... + xm * dzm]
    db = 1 / num * np.sum(dz)  # db = 1/m * (dz1 + dz2 + ... + dzm)

    # 用字典存储dw和db
    gradients = {"dw": dw,
                 "db": db}
    return gradients, cost


# 4. 反向传播函数,采用梯度下降与优化, 求解使损失函数最小的W和b。
def backward_propagate(w, b, X, Y, iters, learning_rate):
    """
    反向传播是采用梯度下降法使误差函数减小

    :param w: 初始权值
    :param b: 初始偏置
    :param X: 输入数据
    :param Y: 输入数据标签
    :param iters: 训练迭代次数
    :param learning_rate: 学习速率
    :return: 权值w,偏置b,梯度gradients和损失函数cost
    """
    # 存储损失函数的值
    costs = []
    for i in range(iters):
        # 初始化梯度和损失函数的值
        gradients, cost = forward_propagate(w, b, X, Y)

        # 获取偏导。dw为总体样本损失函数对w的导数的均值,db为总体样本损失函数对b导数的均值
        dw = gradients["dw"]
        db = gradients["db"]

        # 更新参数权值w和偏置b
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录损失并输出函数
        costs.append(cost)
        print("The cost in the %d th iteration is %f" % (i, cost))

    gradients = {"dw": dw,
                 "db": db}
    return w, b, gradients, costs


# 5.预测
def predict(w, b, X):
    """
    预测结果分为两部分,第一部分计算输出:A = sigmoid(np.dot(w.T, X) + b)。
    然后判断输出和0.5的大小,大于0.5则为1,反之则为0

    :param w: 训练后的权值
    :param b: 训练后偏置
    :param X: 输入数据
    :return:  预测出输入数据的标签
    """
    # 获取输入数目m
    m = X.shape[1]
    # 存储预测结果
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 神经元输出
    A = sigmoid(np.dot(w.T, X) + b)

    # 开始预测
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


# XOY坐标轴的散点图
def plot_dis_data(x, y, color, graph_name):
    '''
    用散点图显示出点的分布

    :param x:   X坐标
    :param y:   Y坐标
    :param color: 点的颜色
    :param graph_name: 图表的名称
    :return:
    '''
    plt.scatter(x, y, s=15, c=color)
    plt.title(graph_name)
    plt.show()


# XOY坐标轴的线性图
def plot_line_chart(data, xlabel, ylabel, graph_name):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(graph_name)
    plt.show()


if __name__ == "__main__":
    # X为横坐标,Y为纵坐标。 相当于两个属性x1,x2
    X = [0, 1.5, 2, 2, 2.25, 2.8, 3.2, 4]
    Y = [1.5, 3, 4, 3, 0.5, 2.8, 1.35, 4]

    label = [1, 1, 1, 1, 0, 0, 0, 0]

    # 1类为蓝色, 0类为红色
    label_color = ['blue', 'red']
    color = []
    for i in label:
        if i == 1:
            color.append(label_color[0])
        else:
            color.append(label_color[1])
    # pyplot绘制原始数据分布图
    plot_dis_data(X, Y, color, 'Raw Data')

    # 数据归一化
    X = np.array(X)
    Y = np.array(Y)
    X = (X - np.average(X))
    Y = (Y - np.average(Y))
    X = X / X.max()
    Y = Y / Y.max()

    # pyplot绘制归一化之后的数据分布图
    plot_dis_data(X, Y, color, 'Normalization Data')

    data_X = np.vstack((X, Y))
    data_label = np.array([label])

    # 参数设置
    w = []
    b = []
    Y_prediction = []
    iters = 50  # 迭代次数
    learning_rate = 0.5  # 学习率
    w, b = init_param_withZeros(data_X.shape[0])  # 初始化w,b
    # 开始训练
    w, b, gradients, costs = backward_propagate(
        w, b, data_X, data_label, iters, learning_rate)
    Y_prediction = predict(w, b, data_X, data_label)

    # pyplot画"损失函数-迭代次数"的线性图
    plot_line_chart(
        costs,
        'iterations',
        'cost',
        "Learning rate =" +
        str(learning_rate))

    # 测试输入数据
    point = input("Please enter a coordinates:\n")
    # 获取坐标
    x = int(point.split(' ')[0])
    y = int(point.split(' ')[1])
    point_data = np.vstack((x, y))

    point_prediction = predict(w, b, point_data, data_label)
    print("The point is below to", end=" ")
    print(point_prediction[0, 0])
