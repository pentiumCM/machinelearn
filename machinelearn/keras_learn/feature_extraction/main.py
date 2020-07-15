#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : main.py
@Time    : 2020/6/8 14:58
@desc	 : 
'''

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# include_top=False:是否包含卷积上面的分类模块
conv_base.summary()

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

train_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/train'
validation_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/validation'
test_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/test'

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    '''
    特征提取
    :param directory:
    :param sample_count:
    :return:
    '''
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)
