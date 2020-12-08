#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : VGG_Net.py
@Time    : 2020/11/14 12:10
@desc	 : VGG 网络结构
'''

# VGG16：包含了16个隐藏层（13个卷积层和3个全连接层）
# VGG19：包含了19个隐藏层（16个卷积层和3个全连接层）

# VGG缺点：
# 3个全连接层，耗费更多计算资源，最上面三层使用来进行分类的，其余层使用来进行特征提取的
# 全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量
# 最后一个全连接层设定了分类的个数

import numpy as np
import keras
import matplotlib.pyplot as plt
import os
import cv2

from utils.log_history_util import AccuracyHistory

from keras.applications import VGG16

from keras.models import Model
from keras.layers import Dense, Flatten, Dropout

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


def load_vgg_model(input_shape=(224, 224, 3)):
    """
    加载VGG16的网络结构
    :param 数据集的输入尺寸
    :return:
    """
    # include_top：是否包含卷积之上的分类器模块
    vgg_model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=input_shape)
    return vgg_model


def setup_base_model(input_shape):
    """
    搭建主干网络中
    卷积部分使用 vgg16 的卷积部分进行特征提取，
    全连接层使用自定义的Dense，来替换VGG16的全连接层

    :param input_shape: 网络输入的数据集的尺寸
    :return:
    """
    # 获取VGG16的网络结构
    vgg_model = load_vgg_model(input_shape)

    # 提取 VGG16 的卷积模块
    inputs = vgg_model.inputs
    outputs = vgg_model.get_layer('block5_pool').output
    conv_block = Model(inputs=inputs, outputs=outputs)

    # 不调整 VGG16中卷积层的参数
    for layer in conv_block.layers:
        layer.trainable = False

    # 获取卷积模块的输出张量
    conv_block_out = conv_block.output

    model_outputs = Flatten(name='flatten')(conv_block_out)
    model_outputs = Dense(1024, activation='relu', name='fc1')(model_outputs)
    model_outputs = Dense(1024, activation='relu', name='fc2')(model_outputs)
    model_outputs = Dropout(0.5)(model_outputs)
    model_outputs = Dense(10, activation='softmax')(model_outputs)  # model_outputs就是最后的y

    base_model = Model(inputs=conv_block.inputs, outputs=model_outputs)

    return base_model


def compile_model(model):
    """
    编译模型
    :param model:
    :return:
    """
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])
    return model


def pre_train_data(input_shape):
    """
    准备训练的数据
    :param input_shape: 训练数据的尺寸
    :return:
    """
    # 数据集的数据
    path = 'F:/develop_code/python/machinelearn/docs/dataset/mnist.npz'

    f = np.load(path)

    # X_train表示训练数据集，总共60,000张28x28大小的手写体图像，y_train表示训练图像对应的标签。
    # 同理，X_test表示测试数据集，总共10,000张28x28大小的手写体图像，y_test表示测试图像对应的标签
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

    x_train = [cv2.cvtColor(cv2.resize(i, (input_shape[0], input_shape[1])), cv2.COLOR_GRAY2RGB)
               for i in x_train]  # 变成彩色的

    x_test = [cv2.cvtColor(cv2.resize(i, (input_shape[0], input_shape[1])), cv2.COLOR_GRAY2RGB)
              for i in x_test]  # 变成彩色的

    # np.concatenate拼接到一起把
    x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')
    x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

    # 使用matplotlib 库将数据可视化
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(x_train[i], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(y_train[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # reshape(samples, features)
    x_train = x_train.reshape(60000, input_shape[0], input_shape[1], input_shape[2])
    x_test = x_test.reshape(10000, input_shape[0], input_shape[1], input_shape[2])
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 将图片灰度值归一化到0~1之间的值
    x_train /= 255
    x_test /= 255

    # 类别进行编码
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    # 参数设置
    # 训练的参数
    num_category = 10
    epochs = 30
    batch_size = 128
    model_path = 'mnist_vgg16.h5'

    input_shape = (48, 48, 3)

    # 加载训练和测试数据
    x_train, y_train, x_test, y_test = pre_train_data(input_shape)

    base_model = setup_base_model(input_shape=input_shape)

    base_model = compile_model(base_model)
    base_model.summary()

    # 模型训练

    history = AccuracyHistory()
    base_model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test),
                   callbacks=[history])

    # 保存模型文件
    if os.path.exists(model_path) == False:
        base_model.save(model_path)

    # 模型训练好后需要评估其性能：
    score = base_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])  # Test loss: 0.0296396646054
    print('Test accuracy:', score[1])  # Test accuracy: 0.9904

    # # 加载测试图片
    # img_path = 'dog.jpg'
    # img = image.load_img(img_path, target_size=(224, 224))
    #
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    #
    # # 预处理图片
    # x = preprocess_input(x)
    #
    # # 预测
    # features = base_model.predict(x)
    # print(features)
    #
    # # 解析预测结果
    # labels = decode_predictions(features)
    # print(labels)
