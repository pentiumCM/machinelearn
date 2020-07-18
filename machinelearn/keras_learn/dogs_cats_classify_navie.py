#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : dogs_cats_classify_navie.py
@Time    : 2020/6/8 20:37
@desc	 : 猫狗分类 - 分类器
'''

import os, shutil

# 下载 dogs-vs-cats 数据集并解压 train.zip
original_dataset_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/dogs-vs-cats/train/'

# 设置数据集的各层级目录
base_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data'
data_dirs = ['train', 'validation', 'test']
class_dirs = ['cat', 'dog']

from keras import models, layers, optimizers

model = models.Sequential()

# 卷积层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# 最大池化层
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

# 全连接层
model.add(layers.Dense(512, activation='relu'))

# sigmoid ： 二分类
model.add(layers.Dense(1, activation='sigmoid'))

# 模型可视化
# from keras.utils import plot_model
# plot_model(model,to_file='model.png')

model.summary()

# 模型编译
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# 数据预处理
# 1. 读取图像文件
# 2. 将JPEG文件解码为RGB像素网格
# 3. 将这些网格转换为浮点数张量
# 4. 将像素值(0 ~ 255)缩放到[0,1]区间

from keras.preprocessing.image import ImageDataGenerator

# 使用ImageDataGenerator 从目录中读取图像
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 创建训练集数据生成器，猫和狗各1000张
train_generator = train_datagen.flow_from_directory(
    'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# 创建验证集数据生成器，猫和狗各500张
test_generator = test_datagen.flow_from_directory(
    'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/test',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print(f'data_batch. shape: {data_batch.shape}')
    print(f'labels_batch.shape: {labels_batch.shape}')
    break

# 模型训练
history = model.fit_generator(train_generator,
                              steps_per_epoch=50,
                              epochs=30,
                              validation_data=test_generator,
                              validation_steps=50)

# 模型保存
model.save('cats_and_dogs_small1.h5')

# 模型评估
import matplotlib.pyplot as plt

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
