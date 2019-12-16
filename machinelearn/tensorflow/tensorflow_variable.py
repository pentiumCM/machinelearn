#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : tensorflow_variable.py
@Time    : 2019/11/14 20:56
@desc	 : tensorflow的Variable用法
'''

# 在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。
# 定义语法： state = tf.Variable()

import tensorflow as tf

state = tf.Variable(0, name='counter')
# 定义常量 one
one = tf.constant(1)

# 定义加法步骤 (注: 此步并没有直接计算)
new_value = tf.add(state, one)

# 将 State 更新成 new_value
update = tf.assign(state, new_value)

# 在 Tensorflow 中设定了变量，那么初始化变量是最重要的！！所以定义了变量以后, 一定要定义 init = tf.initialize_all_variables() .
init = tf.global_variables_initializer()  # 替换成这样就好

# 到这里变量还是没有被激活，需要再在 sess 里, sess.run(init) , 激活 init 这一步.

# 使用 Session
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
