#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/7/22 15:37
@desc	 : 
'''
import cv2
import numpy as np

from keras import backend as K, Input
from keras.applications import VGG16, ResNet50

from keras.layers import Concatenate, Add

from cv_learn.resnet import ResNet50_iml


def CMIANet_fuse_early_layer(feature_map_rgb=None, feature_map_lwir=None, fuse_type='Concatenate'):
    """
    CMIA 融合部分：早期融合，直接叠加通道
    :param input_tensor_rgb:
    :param input_tensor_lwir:
    :param fuse_type:融合方式，Concatenate / add
    :return:
    """

    # 融合RGB与红外特征
    # x = keras.layers.Concatenate(axis=1)([r1, r2])

    if fuse_type == 'Concatenate':
        fuse_layer = Concatenate(name='fm_fuse')([feature_map_rgb, feature_map_lwir])
    elif fuse_type == 'Add':
        fuse_layer = Add(name='fm_fuse')([feature_map_rgb, feature_map_lwir])

    # # attention机制，对卷积层做attention
    # model_out = attention_model(fuse_layer)

    return fuse_layer


if __name__ == '__main__':
    img_rgb = cv2.imread('aaa_visible.png')
    img_rgb = cv2.resize(img_rgb, (300, 300))
    cv2.imshow('image', img_rgb)
    cv2.waitKey(0)

    x_rgb = np.expand_dims(img_rgb, axis=0)
    x_rgb = K.constant(x_rgb)

    input = Input(shape=(300, 300, 3))
    resnet = ResNet50_iml(inputs=input)

    vgg_ = VGG16(include_top=False, weights='imagenet', input_shape=(600, 600, 3))

    print('')
