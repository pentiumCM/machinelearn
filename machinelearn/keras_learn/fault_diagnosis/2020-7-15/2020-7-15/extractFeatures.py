#用来获取由CNN抽取的特征值,由于TSNE_CNN.py中绘图的输入数据
#将获得的特征值存入到draw\dense1_data.npy
#通过np.load("flatten_data")访问数据

from keras import backend as K
from dataProprecess import bearingPro
import numpy as np
from keras.models import load_model


length = 1024  # 每个样本的长度
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例

path = r'F:\develop_code\py_code\today\data\bearing\0hp\0.007'
train_x, train_y, valid_x, valid_y, test_x, test_y = bearingPro.prepro(d_path=path, length=length,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=True, enc_step=28)
# 输入卷积的时候还需要修改一下，增加通道数目
x_train, x_valid, x_test = train_x[:, :, np.newaxis], valid_x[:, :, np.newaxis], test_x[:, :, np.newaxis]

model = load_model(r"model\1DLSTM.h5")

# layer_name = 'dense_1'
#输出各层的信息
# for layer in model_opt1.layers:
#
#        print("{} output shape: {}".format(layer.name, layer.output_shape))
#        print(layer.output)  #获取
#        break
get_dense1_output = K.function([model.layers[0].input], [model.get_layer('dense_1').output])
dense1_layer_output = get_dense1_output([x_train])[0]     #不带[0]的数据格式<1,7000,100>
print(dense1_layer_output)
print(dense1_layer_output.shape)  #<7000,100>

np.save(r"feature\dense1_data.npy",dense1_layer_output)
#使用np.load("flatten_data")加载数据

