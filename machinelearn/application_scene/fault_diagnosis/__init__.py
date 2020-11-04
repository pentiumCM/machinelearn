#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/7/11 13:49
@desc	 : 
'''

import keras

from keras import Sequential
from keras.layers import Permute

model = Sequential()
model.add(Permute(dims=(2, 1), input_shape=(10, 64)))
# model.add(Permute((2, 1), input_shape=(10, 64)))
# 现在: model.output_shape == (None, 64,10)
# “None"是batch的维度

print(model.output_shape)
