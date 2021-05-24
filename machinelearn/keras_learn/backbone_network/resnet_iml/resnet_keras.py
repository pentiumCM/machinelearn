#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : resnet_keras.py
@Time    : 2021/1/21 14:39
@desc	 : 
'''

# from keras_applications.vgg16 import VGG16
from keras.applications import ResNet50
from keras_applications.vgg16 import VGG16

from keras.utils import plot_model

base_layer = ResNet50(input_shape=(300, 300, 3), include_top=False)

base_layer.summary()

plot_model(base_layer, 'resnet.png')
