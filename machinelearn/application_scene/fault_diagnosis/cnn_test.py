#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : cnn_test.py
@Time    : 2020/7/11 20:08
@desc	 : cnn融合特征测试
'''

import numpy as np

import application_scene.fault_diagnosis.bearingPro as bearingPro

from keras.models import load_model

from livelossplot.keras import PlotLossesCallback

# 导入ReduceLROnPlateau - 学习率阶层性下降
from keras.callbacks import ReduceLROnPlateau

# 加载模型结构
model = load_model('./cnn_model.h5')
print(model.summary())

# 训练参数
batch_size = 128
epochs = 50
num_classes = 10  # 共有十种分类
length = 1024  # 每个样本的长度
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例

# 加载数据
path = r'F:\develop_code\python\machinelearn\machinelearn\keras_learn\fault_diagnosis\data'
train_x, train_y, valid_x, valid_y, test_x, test_y = bearingPro.prepro(d_path=path,
                                                                       length=length,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=True,
                                                                       enc_step=28)
# 输入卷积的时候还需要修改一下, 增加通道数目
x_train, x_valid, x_test = train_x[:, :, np.newaxis], valid_x[:, :, np.newaxis], test_x[:, :, np.newaxis]

print('训练样本维度:', x_train.shape)  # (7000, 1024, 1)
print('验证样本的维度', x_valid.shape)  # (2000, 1024, 1)
print('测试样本的维度', x_test.shape)  # (1000, 1024, 1)

# reshape训练集数据，输入到网络结构中
x_train_1D = np.reshape(x_train, (x_train.shape[0], 1024, 1))
x_train_2D = np.reshape(x_train, (x_train.shape[0], 32, 32, 1))

x_val_1D = np.reshape(x_valid, (x_valid.shape[0], 1024, 1))
x_val_2D = np.reshape(x_valid, (x_valid.shape[0], 32, 32, 1))

x_test_1D = np.reshape(x_test, (x_test.shape[0], 1024, 1))
x_test_2D = np.reshape(x_test, (x_test.shape[0], 32, 32, 1))

# 开始模型训练
# 定义ReduceLROnPlateau - 学习率递减
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

model.fit([x_train_1D, x_train_2D], train_y,
          validation_data=([x_val_1D, x_val_2D], valid_y),
          verbose=1,
          epochs=epochs,
          batch_size=batch_size,
          shuffle=True,
          callbacks=[PlotLossesCallback()])

# 评估模型
score = model.evaluate(x=[x_test_1D, x_test_2D], y=test_y, verbose=0)
print("测试集上的损失率：", score[0])
print("测试集上的准确率：", score[1])
