#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : network.py
@Time    : 2019/10/27 21:49
@desc	 : 
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


# 2.初始化参数
def initializeParamWithZeros(dim):
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


# 3.BP算法
def BackPropagate(w, b, X, Y):
    """
    BP算法
    :param w: 权重
    :param b: 偏置
    :param X: 输入数据
    :param Y: 输入标签
    :return: 梯度和损失函数
    """
    # 输入数据数目
    num = Y.shape[0]

    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1 / num * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # 反向传播
    dw = 1 / num * np.dot(X, (A - Y).T)
    db = 1 / num * np.sum(A - Y)

    # 用字典存储dw和db
    gradients = {"dw": dw,
                 "db": db}
    return gradients, cost


# 4. 梯度下降与优化
def Optimize(w, b, X, Y, iters, learning_rate):
    """
    求解最优的W和b

    :param w: 初始权值
    :param b: 初始偏置
    :param X: 输入数据
    :param Y: 输入数据标签
    :param iters: 训练迭代次数
    :param learning_rate: 学习速率
    :return: 权值w，偏置b，梯度gradients和损失函数cost
    """
    # 存储损失函数的值
    costs = []
    for i in range(iters):
        # 初始化梯度和损失函数的值
        gradients, cost = BackPropagate(w, b, X, Y)

        # 获取偏导
        dw = gradients["dw"]
        db = gradients["db"]

        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录损失并输出函数
        costs.append(cost)
        print("The cost in the %d th iteration is %f" % (i, cost))

    gradients = {"dw": dw,
                 "db": db}
    return w, b, gradients, costs


# 5.预测
def predict(w, b, X, Y):
    """
    预测结果分为两部分，第一部分计算输出：
    然后判断输出和0.5的大小，大于0.5则为1，反之则为0

    :param w: 训练后的权值
    :param b: 训练后偏置
    :param X: 输入数据
    :param Y: 输入数据的标签
    :return: 预测概率大于0.5的
    """
    # 获取输入数目m
    m = Y.shape[0]
    # 存储预测结果
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    # 神经元输出
    A = sigmoid(np.dot(w.T, X) + b)

    # 开始预测
    for i in range(A.shape[0]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction


if __name__ == "__main__":
    # X为横坐标，Y为纵坐标
    X = [0, 1.5, 2, 2, 2.25, 2.8, 3.2, 4]
    Y = [1.5, 3, 4, 3, 0.5, 2.8, 1.35, 4]
    label = [1, 1, 1, 1, 0, 0, 0, 0]

    # 第一类为蓝色，第二类为红色
    label_color = ['blue', 'red']
    color = []
    for i in label:
        if i == 1:
            color.append(label_color[0])
        else:
            color.append(label_color[1])
    for i in range(len(X)):
        plt.scatter(X[i], Y[i], c=color[i])
    plt.title('Raw Data')
    plt.show()

    # 数据归一化
    X = np.array(X)
    Y = np.array(Y)
    X = (X - np.average(X))
    Y = (Y - np.average(Y))
    X = X / X.max()
    Y = Y / Y.max()
    for i in range(len(X)):
        plt.scatter(X[i], Y[i], c=color[i])
    plt.title('Normalization Data')
    plt.show()

    data_X = np.vstack((X, Y))
    data_label = np.array([label])
    # 参数设置
    w = []
    b = []
    Y_prediction = []
    iters = 20
    learning_rate = 0.75
    # 开始训练
    w, b = initializeParamWithZeros(data_X.shape[0])
    w, b, gradients, costs = Optimize(w, b, data_X, data_label, iters, learning_rate)
    Y_prediction = predict(w, b, data_X)

    # 画图
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # 测试输入数据
    point = input("Please enter a coordinates:\n")
    # 获取坐标
    x = int(point.split(' ')[0])
    y = int(point.split(' ')[1])
    point_data = np.vstack((x, y))
    point_prediction = predict(w, b, point_data)
    print("The point is below to", end=" ")
    print(point_prediction[0, 0])
