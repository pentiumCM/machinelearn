#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : multimodel_utils.py
@Time    : 2020/11/3 10:35
@desc	 : keras 的多网络结构融合
'''

import keras
from keras.models import Model
from keras.layers import Input, Dense, PReLU, Dropout  # PRelU为带参数的ReLU


def model1(inp):
    """
    分支网络一
    :param inp:
    :return:
    """
    x1 = keras.layers.Dense(8, activation='relu')(inp)
    model = Model(input=inp, outputs=x1)
    return model


def model2(inp2):
    """
    分支网络二
    :param inp2:
    :return:
    """
    x2 = keras.layers.Dense(8, activation='relu')(inp2)
    model = Model(input=inp2, outputs=x2)
    return model


# def merge_model(inp1, inp2):
#     # inp1 = Input(shape=(10, 4))
#     # inp2 = Input(shape=(16, 4))
#     model_1 = model1(inp1)
#     model_2 = model2(inp2)
#
#     # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
#     # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)
#
#     r1 = model_1.output
#     r2 = model_2.output
#     x = keras.layers.Concatenate(axis=1)([r1, r2])
#     model = Model(input=[inp1, inp2], outputs=x)
#     return model


def merge_model(model1, model2):
    """
    融合两个网络结构
    :param model1: 网络1
    :param model2: 网络2
    :return:
    """
    input1 = model1.input
    input2 = model2.input

    out1 = model1.output
    out2 = model2.output
    x = keras.layers.Concatenate(axis=1)([out1, out2])
    model = Model(input=[input1, input2], outputs=x)
    return model


if __name__ == "__main__":
    inp1 = Input(shape=(10, 4))
    inp2 = Input(shape=(16, 4))

    model_1 = model1(inp1)
    model_1.summary()

    model_2 = model2(inp2)
    model_2.summary()

    model_merge = merge_model(model_1, model_2)
    model_merge.summary()

    print("hello")
