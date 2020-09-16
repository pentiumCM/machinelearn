#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : feature_extraction_VGG16.py
@Time    : 2020/6/8 14:58
@desc	 : 特征提取 - 从 VGG16 预训练模型中提取特征
'''

from keras.applications import VGG16

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from keras import models, layers, optimizers

import matplotlib.pyplot as plt

# include_top：是否包含卷积之上的分类器模块
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# include_top=False:是否包含卷积上面的分类模块
conv_base.summary()

train_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/train'
validation_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/validation'
test_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/test'

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    '''
    特征提取
    :param directory: 数据集的路径
    :param sample_count: 样本数量
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

# 目前，提取的特征形状为 (samples, 4, 4, 512)。
# 我们要将其输入到dense分类器中，
# 因此，必须先将其形状展平为 (samples, 8192)。
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# 构建模型分类器
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels),
                    verbose=2)

model.save('cats_and_dogs_small_3.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
