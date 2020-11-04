#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : attention_test.py
@Time    : 2020/7/17 0:09
@desc	 : 注意力机制 - demo代码
'''

# 1、导入相关库文件
from application_scene.fault_diagnosis import get_activations, get_data

import numpy as np

np.random.seed(1337)  # for reproducibility

# 2.定义网络
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import *
from keras.layers import Input, Dense, Multiply, Activation

input_dim = 4


def Att(att_dim, inputs, name):
    V = inputs
    QK = Dense(att_dim, bias=None)(inputs)
    QK = Activation("softmax", name=name)(QK)
    MV = Multiply()([V, QK])
    return (MV)


def build_model():
    inputs = Input(shape=(input_dim,))

    atts1 = Att(input_dim, inputs, "attention_vec")

    x = Dense(16)(atts1)
    atts2 = Att(16, x, "attention_vec1")

    output = Dense(1, activation='sigmoid')(atts2)
    model = Model(input=inputs, output=output)
    return model


# 3.训练与作图
if __name__ == '__main__':
    N = 10000
    inputs_1, outputs = get_data(N, input_dim)

    print(inputs_1[:2], outputs[:2])

    m = build_model()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(m.summary())

    m.fit(inputs_1, outputs, epochs=20, batch_size=128, validation_split=0.2)

    testing_inputs_1, testing_outputs = get_data(1, input_dim)

    # Attention vector corresponds to the second matrix.
    # The first one is the Inputs output.
    attention_vector = get_activations(m, testing_inputs_1,
                                       print_shape_only=True,
                                       layer_name='attention_vec')[0].flatten()
    print('attention =', attention_vector)

    # plot part.

    pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',
                                                                   title='Attention Mechanism as '
                                                                         'a function of input'
                                                                         ' dimensions.')
    plt.show()
