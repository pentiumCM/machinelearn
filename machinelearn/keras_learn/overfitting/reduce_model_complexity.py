#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : reduce_model_complexity.py
@Time    : 2020/5/4 22:11
@desc	 : 通过降低模型复杂度来解决过拟合问题
'''

import os
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras import models, layers
from keras import optimizers
from keras.datasets import imdb
from keras.utils import plot_model

model_path = 'F:/develop_code/python/machinelearn/docs/model/keras_model/imdb.h5'

# 1. 加载imdb数据集
dataset_name = 'imdb.npz'
dataset = imdb.load_data(path=dataset_name, num_words=10000)
(train_data, train_labels), (test_data, test_labels) = dataset


# 2. 向量化数据
# 标注成one-hot形式，将训练数据单词出现在次数在前10000个单词的位置标注
def vectorize(seqs, dim=10000):
    ret = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        ret[i, seq] = 1
    return ret


x_train = vectorize(train_data)
x_test = vectorize(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

batch_size = 128
epochs = 20

# 验证集为数据集的前10000个样本
x_val = x_train[:10000]
# 训练集为剩余15000个样本
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# original Model
# 定义网络为 Sequential 顺序模型，它由多个网络层线性堆叠
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# 中间层的神经元个数太多会造成过拟合
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 打印模型的详细信息
print(model.summary())
plot_model(model, to_file='imdb.png')

# 配置学习过程
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_val, y_val))

print(history.history.keys())
history_dict = history.history
m_model_loss = history_dict['val_loss']

# 可视化损失
loss = history_dict['loss']
val_loss = history_dict['val_loss']
num_epochs = range(1, len(loss) + 1)

plt.plot(num_epochs, loss, 'bo', label='Training loss')
plt.plot(num_epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化准确率
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(num_epochs, acc, 'bo', label='Training acc')
plt.plot(num_epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# 给图加图例
plt.legend()
plt.show()

model.save(model_path)

# 2. large Model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
# 中间层的神经元个数太多会造成过拟合
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 打印模型的详细信息
print(model.summary())
plot_model(model, to_file='imdb.png')

# 配置学习过程
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_val, y_val))
history_dict = history.history
l_model_loss = history_dict['val_loss']

# 3. small Model
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
# 中间层的神经元个数太多会造成过拟合
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 打印模型的详细信息
print(model.summary())
plot_model(model, to_file='imdb.png')

# 配置学习过程
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_val, y_val))
history_dict = history.history
s_model_loss = history_dict['val_loss']

# 可视化不同规模网络结构的损失值
plt.plot(num_epochs, s_model_loss, 'g', label='Small Model')
plt.plot(num_epochs, m_model_loss, 'r', label='Original Model')
plt.plot(num_epochs, l_model_loss, 'b', label='Large Model')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')

plt.legend()
plt.xticks(np.arange(0, 21, step=1))
plt.show()

print("imdb")
