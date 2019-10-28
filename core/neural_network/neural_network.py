#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : neural_network.py
@Time    : 2019/10/26 19:30
@desc	 : 基于计算库numpy的最初级的感知机的神经网络模型
'''

import numpy as np


# 1. 激活函数,构建sigmoid(x)函数
def sigmoid(x):
    """
    利用 numpy 进行定义 sigmoid()
    :param x: 输入
    :return:  sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


# 1.1 激活函数的导数
def sigmoid_derivative(x):
    """
    sigmoid激活函数的导数
    :param x: 输入
    :return:  输出函数对应的导数
    """
    return sigmoid(x) * (1 - sigmoid(x))


# 2.初始化参数,权重w和偏置b
def initilize_with_zeros(dim):
    """
    初始化权重w和偏置b
    :param dim: 输入维度
    :return: 返回初始化的w和b
              w:(dim,1)的向量
              b:标量
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


# 3. 前向传播函数
def forward_propagate(w, b, X, Y):
    """
    前向传播函数中，先是通过激活函数直接表示了感知机输出的预测值A，然后通过定义的交叉熵损失函数计算了损失cost，最后根据损失函数计算了权值 w 和偏置 b的梯度.

    :param w: 权值
    :param b: 偏置
    :param X: 样本数据
    :param Y: 样本数据的真实结果
    :return:  梯度结果grads的字典和损失cost
    """
    m = X.shape[1]
    # A = sigmoid(wx +b)
    A = sigmoid(np.dot(w.T, X) + b)
    # 损失函数采用交叉熵损失
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {'dw': dw,
             'db': db
             }

    return grads, cost


# 4. 反向传播函数
def backward_propagate(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    计算每一步的当前损失cost,根据损失cost对权值w进行更新
    先建立了一个损失列表容器，然后将前向传播函数放进去执行迭代操作，计算每一步的当前损失cost和梯度grad，
    利用梯度下降法对权值w进行更新，并用字典封装迭代结束时的参数和梯度进行返回。

    :param w: 初始权值
    :param b: 初始偏置
    :param X: 输入数据
    :param Y: 输入数据标签
    :param num_iterations: 训练迭代次数
    :param learning_rate: 学习率,控制每一轮迭代中的更新步长
    :param print_cost: 是否打印数据
    :return: 权值w，偏置b，梯度gradients和损失函数costs
    """
    # 存储损失函数的值
    costs = []
    for i in range(num_iterations):
        # 初始化梯度和损失函数的值
        grad, cost = forward_propagate(w, b, X, Y)
        # 获取偏导(权值和偏置的梯度项)
        dw = grad['dw']
        db = grad['db']

        # 更新参数w,b
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" % (i, cost))
    params = {"dw": w,
              "db": b
              }
    grads = {"dw": dw,
             "db": db
             }

    return params, grads, costs


# 5.预测函数 predict
def predict(w, b, X):
    """

    :param w: 训练后的权值
    :param b: 训练后偏置
    :param X: 输入数据
    :return:  预测概率大于0
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
        if A[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0
    assert (Y_prediction.shape == (1, m))
    return Y_prediction


# 基于numpy的感知机模型
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros
    w, b = initilize_with_zeros(X_train.shape[0])

    # 梯度下降
    parameters, grads, costs = backward_propagate(w, b, X_train, Y_train, num_iterations, learning_rate,
                                                  print_cost)
    # 从字典"parameters"中检索参数w和b
    w = parameters["w"]
    b = parameters["b"]

    # 预测  训练/测试集样本
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # 打印  训练/测试错误
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


if __name__ == "__main__":
    X_train = [0, 1.5, 2, 2, 2.25, 2.8, 3.2, 4]
    Y_train = [1, 1, 1, 1, 0, 0, 0, 0]

    X_test = [0, 1.5, 2, 2, 2.5, 2.8, 3.2, 4]
    Y_test = [1, 0, 1, 1, 1, 0, 0, 0]
    model(X_train, Y_train, X_test,Y_test)
