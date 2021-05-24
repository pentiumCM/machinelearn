#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : plot_fm.py
@Time    : 2021/1/23 23:38
@desc	 : 
'''

from keras import Input, Model
from keras.applications import VGG16, ResNet50
from pylab import *
import cv2

if __name__ == "__main__":
    img = cv2.imread('aaa_lwir.png')
    img = cv2.resize(img, (600, 600))
    x = np.expand_dims(img, axis=0)

    input_tense = Input(shape=(600, 600, 3))
    net_ = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tense)
    # model = VGG16(include_top=False, weights='imagenet', input_shape=(600, 600, 3))
    net_.summary()

    layer = net_.get_layer(index=100)
    output = layer.output
    # "activation_28"

    dense_result = Model(inputs=net_.input, outputs=output)

    dense_res = dense_result.predict(x)  # 使用predict得到该层结果

    rand_layer = 1

    # 获取某一层的数据：因为原始数据维度是[1,x,x,depths]的，我们仅仅提取某一个depth对应的[x,x]维度的信息
    x_output = dense_res[0, :, :, rand_layer]

    # 获取最大值，然后对该层数据进行归一化之后投影到0-255之间
    max = np.max(x_output)
    print(max, "max value is :")
    # 然后进行归一化操作
    x_output = x_output.astype("float32") / max * 255
    print(x_output.shape)

    # 把图像转换成image可以表示的方式进行显示
    from PIL import Image as PILImage

    x_output = PILImage.fromarray(np.asarray(x_output))
    x_output1 = x_output.resize((400, 400))
    x_output1.show()
    print(np.asarray(x_output1))
