#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : tensorflow_struct.py
@Time    : 2020/3/5 21:29
@desc	 : tensorflow的基本结构
'''

import tensorflow as tf


def tensorflow_demo():
    """
    TensorFlow基本结构
    :return:
    """
    # 原生python加法运算
    a = 2
    b = 3
    c = a + b
    print("普通加法运算结果：", c)

    # Tensorflow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("tensorflow加法运算的结果：", c_t)

    # 开启会话
    with tf.compat.v1.Session() as sess:
        c_t_value = sess.run(c_t)
        print("会话中的值：", c_t_value)

    return None


if __name__ == "__main__":
    # 代码1：TensorFlow基本结构
    tensorflow_demo()
