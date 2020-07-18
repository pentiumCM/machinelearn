#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : attention_mnist.py
@Time    : 2020/7/17 19:47
@desc	 : 
'''

# mnist attention
import numpy as np

np.random.seed(1337)
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

# 可视化神经网络
from keras.utils import plot_model

TIME_STEPS = 28
INPUT_DIM = 28
lstm_units = 64

# data pre-processing
path = 'F:/develop_code/python/machinelearn/docs/dataset/mnist.npz'
f = np.load(path)
# X_train表示训练数据集，总共60,000张28x28大小的手写体图像，y_train表示训练图像对应的标签。
# 同理，X_test表示测试数据集，总共10,000张28x28大小的手写体图像，y_test表示测试图像对应的标签
X_train, y_train = f['x_train'], f['y_train']
X_test, y_test = f['x_test'], f['y_test']

X_train = X_train.reshape(-1, 28, 28) / 255.
X_test = X_test.reshape(-1, 28, 28) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# first way attention
def attention_3d_block(inputs):
    # input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


# build RNN model with attention
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
drop1 = Dropout(0.3)(inputs)
lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
attention_mul = attention_3d_block(lstm_out)
attention_flatten = Flatten()(attention_mul)
drop2 = Dropout(0.3)(attention_flatten)
output = Dense(10, activation='sigmoid')(drop2)
model = Model(inputs=inputs, outputs=output)

# second way attention
# inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
# units = 32
# activations = LSTM(units, return_sequences=True, name='lstm_layer')(inputs)
#
# attention = Dense(1, activation='tanh')(activations)
# attention = Flatten()(attention)
# attention = Activation('softmax')(attention)
# attention = RepeatVector(units)(attention)
# attention = Permute([2, 1], name='attention_vec')(attention)
# attention_mul = merge([activations, attention], mode='mul', name='attention_mul')
# out_attention_mul = Flatten()(attention_mul)
# output = Dense(10, activation='sigmoid')(out_attention_mul)
# model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

plot_model(model=model, to_file='attention_mnist.png', show_shapes=True)
plot_model(model=model, to_file='attention_mnist_1.png')

print('Training------------')
model.fit(X_train, y_train, epochs=10, batch_size=16)

print('Testing--------------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)
