#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : mnist_cnn.py
@Time    : 2020/7/15 11:16
@desc	 : 利用卷积神经网络来实现手写体数字识别
'''

import numpy as np
import matplotlib.pyplot as  plt

import keras

from utils.log_history_util import AccuracyHistory

# 导入相关层的结构
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

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
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 将图片灰度值归一化到0~1之间的值
x_train /= 255
x_test /= 255

# 类别进行编码
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

model = Sequential()
# filters卷积核的数量，即卷积层输出的通道数
# input_shape与输入层的大小提供模型
model.add(Conv2D(filters=32,
                 kernel_size=(5, 5),
                 padding='same',
                 activation='relu',
                 input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

# 卷积输出平坦化，即将结果设置成一维
model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_category, activation='softmax'))

history = AccuracyHistory()
# 编译模型
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

# 可视化模型
plot_model(model, 'mnist_cnn.png')
print(model.summary())

# 模型训练
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

# 模型训练好后需要评估其性能：
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])  # Test loss: 0.0296396646054
print('Test accuracy:', score[1])  # Test accuracy: 0.9904

plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
