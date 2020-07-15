#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : hello_mnist.py
@Time    : 2020/4/11 14:15
@desc	 : keras实现的mnist手写体识别
'''

import numpy as np

# 导入手写体的数据集
from keras.datasets import mnist

# 导入相关层的结构
from keras.models import Sequential
from keras.layers import Dense

# 导入one-hot,标签转化为向量
from keras.utils import to_categorical

# 可视化神经网络
from keras.utils import plot_model

# 导入keras的优化器
from keras.optimizers import RMSprop

model_path = 'F:/develop_code/python/machinelearn/docs/model/keras_model/mnist.h5'

path = 'F:/develop_code/python/machinelearn/docs/dataset/mnist.npz'

# data = mnist.load_data('mnist.npz')

f = np.load(path)

# X_train表示训练数据集，总共60,000张28x28大小的手写体图像，y_train表示训练图像对应的标签。
# 同理，X_test表示测试数据集，总共10,000张28x28大小的手写体图像，y_test表示测试图像对应的标签
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']

# (x_train, y_train), (x_test, y_test) = data

# reshape(samples, features)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 将图片灰度值归一化到0~1之间的值
x_train /= 255
x_test /= 255

# 分类类别
num_category = 10

# one-hot, 将类别标签转化为二进制的向量，第i位为1，就表示为第i类
y_train = to_categorical(y_train, num_category)
y_test = to_categorical(y_test, num_category)

# Sequential 顺序模型，它由多个网络层线性堆叠
model = Sequential()
# 使用 .add() 来堆叠模型：
model.add(Dense(784, activation='relu', input_shape=(784,), name='Dense_0'))
model.add(Dense(512, activation='relu', name='Dense_1'))
model.add(Dense(num_category, activation='softmax', name='Dense_2'))

print(model.summary())

plot_model(model, to_file='mnist_DNN.png')

# 使用 .compile() 来配置学习过程：
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
history = model.fit(x_train, y_train,
                    batch_size=128, epochs=10,
                    verbose=2, validation_data=(x_test, y_test))

# 评估模型性能
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

# 保存训练好的模型
model.save(model_path)
