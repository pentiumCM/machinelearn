#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : build_model.py
@Time    : 2020/8/28 13:58
@desc	 : 搭建和训练网络结构
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# 输入图片的尺寸
img_width, img_height = 18, 36

# nb_train_samples = 30000
# nb_validation_samples = 10000
nb_epoch = 30

# 搭建网络结构
# 卷积层作为首层时需要指明 input_shape 。 使用 TensorFlow 作为 Keras 后台，因此表示单通道的 "1" 应放在图片尺寸后。
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

# sigmoid ： 二分类
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# 由于数据集中目标已经被人工标记并居中，这里不使用任何数据提升。
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# 使用ImageDataGenerator 从目录中读取图像
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# 创建训练集数据生成器，行人和非行人4800和5000
train_generator = train_datagen.flow_from_directory(
    'F:/experiment/data/DC-ped-dataset_base/data/train',
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=128,
    class_mode='binary')

# 创建验证集数据生成器，行人和非行人4800和5000
validation_generator = test_datagen.flow_from_directory(
    'F:/experiment/data/DC-ped-dataset_base/data/test',
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=128,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print(f'data_batch. shape: {data_batch.shape}')
    print(f'labels_batch.shape: {labels_batch.shape}')
    break

# 模型训练
history = model.fit_generator(train_generator,
                              steps_per_epoch=50,
                              epochs=nb_epoch,
                              validation_data=validation_generator,
                              validation_steps=50,
                              verbose=2)

# 模型保存
model.save('first_model.h5')

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
