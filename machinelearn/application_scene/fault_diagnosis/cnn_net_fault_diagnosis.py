#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : cnn_net_fault_diagnosis.py
@Time    : 2020/7/11 13:35
@desc	 : 构建故障检测的 cnn（1D+2D）网格结构
'''

# 导入相关的包
import keras

# 导入相关层的结构
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Dropout
from keras import Model

# 可视化神经网络
from keras.utils import plot_model


def merge_model(model_1, model_2):
    '''
    keras将两个独立的模型融合起来
    :param model_1:
    :param model_2:
    :return:
    '''

    # model_1.load_weights('model_1_weight.h5')#这里可以加载各自权重
    # model_2.load_weights('model_2_weight.h5')#可以是预训练好的模型权重(迁移学习)

    inp1 = model_1.input  # 第一个模型的参数
    inp2 = model_2.input  # 第二个模型的参数
    r1 = model_1.output
    r2 = model_2.output
    x = keras.layers.Concatenate(axis=1)([r1, r2])
    model = Model(inputs=[inp1, inp2], outputs=x)
    return model


def addLayers_model(model):
    '''
    修改模型（模型加层）
    采用 keras 的 Concatenate 进行特征融合之后，模型加层的 add 将无效，所以采用这种方案进行加层
    :param model: 待扩层的模型
    :return:
    '''
    origin_model = model
    for layer in origin_model.layers:
        layer.trainable = False  # 原来的不训练,冻结网络层

    inp = origin_model.input
    x = origin_model.output
    den = Dense(512, name="fine_dense")(x)
    l = Dropout(0.5)(den)
    result = Dense(10, activation="softmax")(l)
    model = Model(input=inp, outputs=result)
    return model


input_shape_1D = (1024, 1)
input_shape_2D = (32, 32, 1)

# 构建模型
# 网络结构（卷积层：relu - 池化层 - 卷积层 - 池化层 - Flatten - 汇聚层 - 全连接层 - Dropout - softmax）
# ====================1、 1D部分 ==============================
model1 = Sequential()
# Conv1D：8 @ 1*1024。8个过滤器（卷积核），卷积核大小设置为3*3
model1.add(Conv1D(filters=8,
                  kernel_size=(3),
                  input_shape=input_shape_1D,
                  padding='same',
                  activation='relu'))

# MaxPooling1D：8 @ 1*512。
model1.add(MaxPooling1D(pool_size=(2), padding='same'))

# Conv1D：16 @ 1*512。16个过滤器，大小设置为3*3
model1.add(Conv1D(filters=16,
                  kernel_size=(3),
                  input_shape=(1, 512),
                  padding='same',
                  activation='relu'))

# MaxPooling1D：16 @ 1*256。
model1.add(MaxPooling1D(pool_size=(2), padding='same'))

model1.add(Flatten())  # 压平：将输出压平为1维
# =============================================================

# ============  ======== 2、 2D部分 ============================
model2 = Sequential()
# Conv2D：8 @ 32*32。8个过滤器（卷积核），卷积核大小设置为3*3
model2.add(Conv2D(filters=8,
                  kernel_size=(3, 3),
                  input_shape=input_shape_2D,
                  padding='same',
                  activation='relu'))

# MaxPooling2D：8 @ 16*16。
model2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Conv2D：16 @ 16*16。16个过滤器，卷积核大小设置为3*3
model2.add(Conv2D(filters=16,
                  kernel_size=(3, 3),
                  input_shape=(16, 16, 1),
                  padding='same',
                  activation='relu'))

# MaxPooling2D：16 @ 8*8。
model2.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model2.add(Flatten())
# =============================================================


# ==================== 3、汇聚层 ===============================
# 融合部分
model = merge_model(model1, model2)
model.summary()
# =============================================================

# ============= 4、 全连接层，dropout，分类层 ====================
model = addLayers_model(model)
print(model.summary())

plot_model(model, to_file='fault_diagnosis.png')
# =============================================================

# ==================== 5、模型训练指标 ==========================
# adam优化器, lr：初始学习率为0.1，学习率下降递减采用：ReduceLROnPlateau，在 model.fit X的回调函数中设置
adam = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
# =============================================================

# 保存模型结构
model.save('./cnn_model.h5')
