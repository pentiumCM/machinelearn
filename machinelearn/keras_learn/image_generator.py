#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : image_generator.py
@Time    : 2020/5/29 15:10
@desc	 : 图像增强测试
'''

# 数据预处理
# 1. 读取图像文件
# 2. 将JPEG文件解码为RGB像素网格
# 3. 将这些网格转换为浮点数张量
# 4. 将像素值(0 ~ 255)缩放到[0,1]区间



from keras.preprocessing.image import ImageDataGenerator

# 使用ImageDataGenerator 从目录中读取图像
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/train'
test_dir = 'F:/develop_tools/anaconda/jupyter-workplace/kaggle/dogs-vs-cats/data/test'

# 创建训练集数据生成器，猫和狗各1000张
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# 创建验证集数据生成器，猫和狗各500张
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

print('hello1')
