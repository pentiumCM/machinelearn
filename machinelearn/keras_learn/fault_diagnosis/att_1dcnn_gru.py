#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : att_1dcnn_gru.py
@Time    : 2020/7/17 19:47
@desc	 :
'''

from keras_learn.fault_diagnosis import bearingPro
import numpy as np
from keras.layers import Input, Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten, LSTM, \
    GRU, GlobalAveragePooling1D, Reshape, Multiply
from keras.models import Sequential, Model
from keras.regularizers import l2  # 正则化器允许在优化过程中对层的参数或层的激活情况进行惩罚
from livelossplot import PlotLossesKeras  # 绘制损失曲线
from keras.utils.vis_utils import plot_model  # 可以直接将搭建的神经网络用流程图表示出来

# 训练参数
batch_size = 128  # 在训练集中选择一组样本用来更新权值，一个batch包含的样本数一般为2的幂次方，常用64，128，256，网络较小的时候选择256，较大时选择64
epochs = 30
num_classes = 10  # 共有十种分类
length = 1024  # 每个样本的长度
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例

# 1、数据准备
path = r'F:\develop_code\python\machinelearn\machinelearn\keras_learn\fault_diagnosis\data'
train_x, train_y, valid_x, valid_y, test_x, test_y = bearingPro.prepro(d_path=path, length=length,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=True, enc_step=28)
# 输入卷积的时候还需要修改一下，增加通道数目
x_train, x_valid, x_test = train_x[:, :, np.newaxis], valid_x[:, :, np.newaxis], test_x[:, :, np.newaxis]
# 输入数据的维度
input_shape = x_train.shape[1:]  # <1024,1>


# 2.1、添加卷积池化模块
def add_con1D_pooling(input, filters_num, kernerl_size, con_strides, conv_padding, is_batchNormal,
                      pool_size, pool_padding, activation):
    """
    往网络结构里面添加卷积池化模块
    :param input: 卷积模块的先前输入
    :param filters_num: 卷积核的数量
    :param kernerl_size: 卷积核的尺寸
    :param con_strides: 卷积的步长
    :param is_batchNormal: 是否Batchnormal，布尔值
    :param pool_size:  池化网格的尺寸
    :param conv_padding: 卷积填充
    :param pool_padding: 池化填充
    :param activation: 卷积核的激活函数
    :return: 返回模型的张量
    """
    # 卷积层
    model = Conv1D(filters=filters_num, kernel_size=kernerl_size,
                   strides=con_strides, padding=conv_padding)(input)
    # BN层
    if is_batchNormal:
        model = BatchNormalization()(model)
    # 卷积层的激活函数
    model = Activation(activation)(model)
    # 池化层
    model = MaxPooling1D(pool_size=pool_size, padding=pool_padding)(model)
    return model


# 2.2、建立卷积层网络结构
def build_conv_block(input):
    """
    建立卷积池化网络结构
    :param input: 模型的初始输入
    :return: 返回卷积池化模块的张量
    """
    model_input = input

    # 卷积池化模块 1
    con_pool1 = add_con1D_pooling(model_input, filters_num=16, kernerl_size=64, con_strides=16,
                                  conv_padding='same', is_batchNormal=BatchNorm,
                                  pool_size=2, pool_padding='same', activation='relu')

    # 卷积池化模块 2
    con_pool2 = add_con1D_pooling(con_pool1, filters_num=32, kernerl_size=3, con_strides=1,
                                  conv_padding='same', is_batchNormal=BatchNorm,
                                  pool_size=2, pool_padding='valid', activation='relu')

    # 卷积池化模块 3
    con_pool3 = add_con1D_pooling(con_pool2, filters_num=64, kernerl_size=3, con_strides=1,
                                  conv_padding='same', is_batchNormal=BatchNorm,
                                  pool_size=2, pool_padding='valid', activation='relu')

    # 卷积池化模块 4
    con_pool4 = add_con1D_pooling(con_pool3, filters_num=64, kernerl_size=3, con_strides=1,
                                  conv_padding='valid', is_batchNormal=BatchNorm,
                                  pool_size=2, pool_padding='valid', activation='relu')
    return con_pool4


# 3、建立GRU层
def build_GRU(input):
    """
    建立 GRU 层
    :param input: GRU层的输入为卷积池化模块的输出
    :return: GRU层的输出
    """
    # gru_out = GRU(units=32, return_sequences=True)(input)
    input_channels = int(input.shape[-1])
    gru_out = GRU(output_dim=input_channels, return_sequences=True)(input)
    return gru_out


# 4、建立注意力机制
def build_attention_model(input):
    """

    :param input: 注意力层的输入为GRU层的输出，本质对卷积的通道赋予不同的权值
    :return: GRU 操作之后的张量
    """
    input_channels = int(input.shape[-1])

    x = GlobalAveragePooling1D()(input)
    x = Dense(int(input_channels / 4))(x)
    x = Activation('softmax')(x)
    x = Dense(input_channels)(x)
    x = Activation('softmax')(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([input, x])
    return x


# 5、搭建整体网路结构：con1D - GRU - attention
def att_1dcnn_gru(input_shape):
    """

    :param model_input: 网络结构输入，张量
    :return:
    """

    model_input = Input(input_shape)

    # 构建卷积池化模块
    model_out = build_conv_block(model_input)

    # Dropout
    model_out = Dropout(0.25)(model_out)

    # GRU层, GRU对卷积之后的 feature Map(时间序列)继续做一次特征提取
    model_out = build_GRU(model_out)

    # attention机制，对卷积层做attention
    model_out = build_attention_model(model_out)

    # 神经元展开
    model_out = Flatten()(model_out)

    # 全连接层
    model_out = Dense(100, activation='relu', kernel_regularizer=l2(1e-4))(model_out)
    model_out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(1e-4))(model_out)

    model = Model(inputs=model_input, outputs=model_out)
    model.summary()
    plot_model(model, 'tensor_model.png', show_shapes=True)

    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 模型训练
    model.fit(x=x_train, y=train_y, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_valid, valid_y), shuffle=True, callbacks=[PlotLossesKeras()])

    # 评估模型
    score = model.evaluate(x=x_test, y=test_y, verbose=0)
    print("Loss on test set：", score[0])
    print("Accuracy on test set:", score[1])

    # 保存模型
    # model.save(r"F:\develop_code\py_code\today\OPT1\model_opt1\CNNmain_opt1.h5")
    # print("保存成功")

    # predict = model_opt1.predict_classes(x_test)
    # print(predict)


if __name__ == '__main__':
    input_shape = (1024, 1)
    att_1dcnn_gru(input_shape)
