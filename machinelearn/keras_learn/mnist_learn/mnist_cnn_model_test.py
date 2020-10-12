#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : mnist_cnn_model_test.py
@Time    : 2020/8/19 15:27
@desc	 : mnist_test-cnn 模型应用 - 使用 mnist 模型进行手写体预测
'''

import numpy as np
from keras.models import load_model

import cv2
import matplotlib.pyplot as plt

from utils.io_utils import get_subfiles

# 待预测图片路径
img_dir = 'F:/develop_code/python/machinelearn/docs/dataset/test/mnist_test/'
# 训练好的模型路径
model_path = 'F:/develop_code/python/machinelearn/docs/model/keras_model/mnist_cnn.h5'

test_files = get_subfiles(img_dir)

for i in range(len(test_files)):
    img_path = img_dir + test_files[i]
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test_pic', src)
    cv2.waitKey(0)

    src = cv2.resize(src, (28, 28))

    # reshape(samples, features)
    x_train = src.reshape(1, 28, 28, 1)
    x_train = x_train.astype('float32')

    # 将图片灰度值归一化到0~1之间的值
    x_train /= 255

    # 加载整个模型
    model = load_model(model_path)

    preds = model.predict(x_train)

    print('Output of the softmax layer:', preds[0])
    print('Network prediction:', np.argmax([preds[0]]))
