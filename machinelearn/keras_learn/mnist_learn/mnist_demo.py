#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : mnist_demo.py
@Time    : 2020/2/25 23:45
@desc	 : 
'''

# 源代码不能直接下载，在这里进行稍微修改，下载数据集后指定路径
# 下载链接：https://pan.baidu.com/s/1jH6uFFC 密码: dw3d

from __future__ import print_function
import keras
import numpy as np

import matplotlib.pyplot as plt

# 导入相关层的结构
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k

# 准备数据
# 加载数据集
path = 'F:/develop_code/python/machinelearn/docs/dataset/mnist_test.npz'
mode_path = 'F:/develop_code/python/machinelearn/docs/model/'
f = np.load(path)

# X_train表示训练数据集，总共60,000张28x28大小的手写体图像，y_train表示训练图像对应的标签。
# 同理，X_test表示测试数据集，总共10,000张28x28大小的手写体图像，y_test表示测试图像对应的标签
X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test']

# 使用matplotlib 库可以将数据可视化
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# 对于神经网络而言，一般需要对原始数据进行预处理。
# 常见的预处理方式是调整图像大小、对像素值进行归一化等。
# let's print the actual data shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# input image size 28*28
img_rows, img_cols = 28, 28

# reshaping 调整维度
# "channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
# more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)  # X_train shape: (60000, 28, 28, 1)

# 标签数据y_train和y_test被转换为分类格式（向量形式），即标签‘3’被转换为向量[ 0,0,0,1,0,0,0,0,0,0]用于建模，
# 标签向量非零的位置减一（从0开始）后表示该图像的具体标签，即若图像的标签向量在下标5处不为0，则表示该图像代表数字‘4’。

# 类别进行编码
# set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

# 构建和编译模型
# 定义模型的体系结构并使用必要的优化函数，损失函数和性能指标进行编译
# 构建模型遵循的体系结构是经典卷积神经网络，分别含有2个卷积层，之后是连接全连接层和softmax分类器
model = Sequential()
# convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 32 convolution filters used each of size 3x3
# again
model.add(Conv2D(64, (3, 3), activation='relu'))
# 64 convolution filters used each of size 3x3

# 在最大池化层和全连接层之后，模型中引入dropout作为正则化来减少过拟合问题
# choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))
# flatten since too many dimensions, we only want a classification output
model.add(Flatten())
# fully connected to get all relevant data
model.add(Dense(128, activation='relu'))
# one more dropout for convergence' sake :)
model.add(Dropout(0.5))
# output a softmax to squash the matrix into output probabilities
model.add(Dense(num_category, activation='softmax'))

# 模型搭建好之后，需要进行编译。
# 在本文使用categorical_crossentropy多分类损失函数。
# 由于所有的标签都具有相似的权重，因此将其作为性能指标，并使用AdaDelta梯度下降技术来优化模型参数。
# Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
# categorical ce since we have multiple classes (10)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 训练和评估模型
# Epoch表示对所有训练样本进行一个前向传播过程和一个反向传播过程，Batch_Size表示每次前向过程和反向过程时处理的训练样本数
batch_size = 128
num_epoch = 10
# model training
model_log = model.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=num_epoch,
                      verbose=1,
                      validation_data=(X_test, y_test))

# 模型训练好后需要评估其性能：
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])  # Test loss: 0.0296396646054
print('Test accuracy:', score[1])  # Test accuracy: 0.9904

# 保存模型参数
# Save the model
# serialize model to JSON
model_digit_json = model.to_json()
with open(mode_path + "model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
model.save_weights(mode_path + "model_digit.h5")
print("Saved model to disk")
