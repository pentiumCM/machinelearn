#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : tensorflow_demo1.py
@Time    : 2019/11/14 17:31
@desc	 : tensorflow的示例demo
'''

import tensorflow as tf
import numpy as np

# 使用 numpy 来创建数据.
x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1 + 0.3 想象成 y=Weights * x + biases,然后神经网络也就是学着把 Weights 变成 0.1, biases 变成 0.3.
y_data = x_data * 0.1 + 0.3

### creat tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
# 计算误差,接着就是计算 y 和 y_data 的误差:
loss = tf.reduce_mean(tf.square(y - y_data))
# 反向传递误差的工作就教给optimizer了, 我们使用的误差传递方法是梯度下降法: Gradient Descent 使用 optimizer 来进行参数的更新.
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 在使用这个结构之前, 我们必须先初始化所有之前定义的Variable
init = tf.global_variables_initializer()  # 替换成这样就好

### creat tensorflow structure end ###

# 创建会话 Session,用 Session 来执行 init 初始化步骤. 并且, 用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
sess = tf.Session()
sess.run(init)  # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
