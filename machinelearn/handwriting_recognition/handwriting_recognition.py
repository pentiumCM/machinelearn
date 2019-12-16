#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : handwriting_recognition.py
@Time    : 2019/10/23 13:14
@desc	 : Tensorflow卷积神经网络（CNN）手写数字识别
'''

import tensorflow as tf
import docs.conf as conf

# 加载MNIST数据集
import tensorflow.examples.tutorials.mnist.input_data as input_data

# 执行read_data_sets()函数后将会返回一个DataSet实例，其中包含训练数据、验证数据和测试数据。
mnist = input_data.read_data_sets(conf.handwriting_recognition_dataset_dir, one_hot=True)

# 创建一个交互式的Session。
sess = tf.InteractiveSession()

# 创建Session和占位符
# x和y_都是tensor，其中x表示输入数据，由于是28×28像素的灰度图片，因此输入为784维的向量。
# y_表示模型输出，为0-9的数字，因此是10维的向量。
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


# 通过函数的形式定义权重变量。变量的初始值，来自于截取正态分布中的数据。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 通过函数的形式定义偏置量变量，偏置的初始值都是0.1，形状由shape定义。
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积函数，其中x是输入，W是权重，也可以理解成卷积核，strides表示步长，或者说是滑动速率，包含长宽方向的步长。
# padding表示补齐数据。 目前有两种补齐方式，一种是SAME，表示补齐操作后（在原始图像周围补充0），实际卷积中，参与计算的原始图像数据都会参与。
# 一种是VALID，补齐操作后，进行卷积过程中，原始图片中右边或者底部的像素数据可能出现丢弃的情况。
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 这步定义函数进行池化操作，在卷积运算中，是一种数据下采样的操作，降低数据量，聚类数据的有效手段。
# 常见的池化操作包含最大值池化和均值池化。这里的2*2池化，就是每4个值中取一个，池化操作的数据区域边缘不重叠。
# 函数原型：def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)。对ksize和strides
# 定义的理解要基于data_format进行。默认NHWC，表示4维数据，[batch,height,width,channels]. 下面函数中的ksize，
# strides中，每次处理都是一张图片，对应的处理数据是一个通道（例如，只是黑白图片）。长宽都是2，表明是2*2的池化区域，也反应出下采样的速度。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义卷积层1的权重和偏置量
W_conv1 = weight_variable([5, 5, 1, 32])
# 偏置量定义，偏置的维度是32.
b_conv1 = bias_variable([32])

# 一、卷积层1
# 将输入tensor x 调整成为28×28矩阵形式。
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 进行卷积操作W × X + b，得到线性变化的结果r_conv1，再利用Tensorflow的relu规则进行非线性映射，出的卷积的结果h_conv1。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积，卷积核大小5*5，输入通道有32个，输出通道有64个，从输出通道数看，第二层的卷积单元有64个。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# 第二层卷积：激活和池化（类似第一层卷积操作的激活和池化）
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层
# 图片尺寸减小到7x7，加入一个有1024个神经元的全连接层，用于处理整个图片。
# 把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU激活操作。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# 二、池化层1
# 将第二层池化后的数据进行变形
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 进行矩阵乘，加偏置后进行relu激活
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
# 对第二层卷积经过relu后的结果，基于tensor值keep_prob进行保留或者丢弃相关维度上的数据。这个是为了防止过拟合，快速收敛。
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
# 输出层为10维的向量，通过softmax函数输出。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 最后，添加一个softmax层，就像前面的单层softmax regression一样。softmax是一个多选择分类函数，其作用和sigmoid这个2值
# 分类作用地位一样，在我们这个例子里面，softmax输出是10个。
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 实际值y_与预测值y_conv的自然对数求乘积，在对应的维度上上求和，该值作为梯度下降法的输入
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

# 下面基于步长1e-4来求梯度，梯度下降方法为AdamOptimizer。
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 首先分别在训练值y_conv以及实际标签值y_的第一个轴向取最大值，比较是否相等
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 对correct_prediction值进行浮点化转换，然后求均值，得到精度。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 先通过tf执行全局变量的初始化，然后启用session运行图。
sess.run(tf.global_variables_initializer())
for i in range(1000):
    # 从mnist的train数据集中取出50批数据，返回的batch其实是一个列表，元素0表示图像数据，元素1表示标签值
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        # 计算精度，通过所取的batch中的图像数据以及标签值还有dropout参数，带入到accuracy定义时所涉及到的相关变量中，进行
        # session的运算，得到一个输出，也就是通过已知的训练图片数据和标签值进行似然估计，然后基于梯度下降，进行权值训练。
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    # 此步主要是用来训练W和bias用的。基于似然估计函数进行梯度下降，收敛后，就等于W和bias都训练好了。
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 对测试图片和测试标签值以及给定的keep_prob进行feed操作，进行计算求出识别率。就相当于前面训练好的W和bias作为已知参数。
print("cf accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
