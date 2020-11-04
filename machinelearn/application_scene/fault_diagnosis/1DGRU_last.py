from application_scene.fault_diagnosis import bearingPro
import numpy as np
from keras.layers import Conv1D, Dense, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.regularizers import l2  # 正则化器允许在优化过程中对层的参数或层的激活情况进行惩罚
from livelossplot import PlotLossesKeras  # 绘制损失曲线
from keras.utils.vis_utils import plot_model  # 可以直接将搭建的神经网络用流程图表示出来

# 训练参数
batch_size = 128  # 在训练集中选择一组样本用来更新权值，一个batch包含的样本数一般为2的幂次方，常用64，128，256，网络较小的时候选择256，较大时选择64
epochs = 10
num_classes = 10  # 共有十种分类
length = 1024  # 每个样本的长度
BatchNorm = True  # 是否批量归一化
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例

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


# 定义卷积层
def addLayers(filters, kernerl_size, strides, conv_padding, pool_padding, pool_size, BatchNormal):
    """
    :param filters: 卷积核的数目，整数
    :param kernerl_size: 卷积核的尺寸，整数
    :param strides: 步长，整数
    :param conv_padding: 'same','valid'
    :param pool_padding: 'same','valid'
    :param pool_size: 池化层核尺寸，整数
    :param BatchNormal: 是否Batchnormal，布尔值
    :return: model_opt1
    """
    model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
                     padding=conv_padding, kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))
    return model


# 实例化序贯模型
model = Sequential()
# 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来，其中kernel_regularizer是正则化，减少模型的过拟合
model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4),
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))  # 对信号进行最大值池化

# 第二层卷积

model = addLayers(filters=32, kernerl_size=3, strides=1, conv_padding='same',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 第三层卷积
model = addLayers(filters=64, kernerl_size=3, strides=1, conv_padding='same',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)

# 第四层卷积
model = addLayers(filters=64, kernerl_size=3, strides=1, conv_padding='valid',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 从卷积到全连接需要展平，Flatten()常用于卷积层与全连接层的过渡
# model.add(GRU(32, return_sequences=True))
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))

# 增加输出层即Softmax层
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

# 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

plot_model(model=model, to_file='CNN_GRU.png', show_shapes=True)

# 开始模型训练
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
