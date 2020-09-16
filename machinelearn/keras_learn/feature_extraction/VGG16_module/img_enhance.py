#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : img_enhance.py
@Time    : 2020/9/4 10:28
@desc	 : 使用数据增强的特征提取
'''

from keras import models
from keras import layers

from keras.applications import VGG16

# include_top：是否包含卷积之上的分类器模块
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

conv_base.trainable = False
