#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : test.py
@Time    : 2021/5/12 23:45
@desc	 : non local测试代码
'''

import sys

sys.path.append('F:\develop_code\python\machinelearn\machinelearn')
path = sys.path
print(path)

from keras_learn.trick.non_local.non_local import non_local_block
from keras import Input, Model
from keras.layers import Conv2D
from keras.utils import plot_model

ip = Input(shape=(640, 480, 3), name='input')

# x = ConvND(...)  # as againm can be Conv1D, Conv2D or Conv3D

x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same', name='conv1')(ip)

x = non_local_block(x, compression=2, mode='embedded')

model = Model(inputs=ip, outputs=x)

model.summary()

plot_model(model=model, to_file='model.png', show_shapes=True)
