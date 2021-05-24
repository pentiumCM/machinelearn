#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : VGG16_net_iml.py
@Time    : 2021/1/19 21:43
@desc	 : 
'''
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, K, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D


class VGG16_net_iml():

    def __init__(self, include_top=True, input_tensor=None, input_shape=None, classes=1000, pooling=None):
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.classes = classes
        self.pooling = pooling

    def vgg16_iml(self):
        if self.input_tensor is None:
            img_input = Input(shape=self.input_shape)
        else:
            if not K.is_keras_tensor(self.input_tensor):
                img_input = Input(tensor=self.input_tensor, shape=self.input_shape)
            else:
                img_input = self.input_tensor

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if self.include_top:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(self.classes, activation='softmax', name='predictions')(x)
        else:
            if self.pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Create model.
        model = Model(img_input, x, name='vgg16')

        return model
