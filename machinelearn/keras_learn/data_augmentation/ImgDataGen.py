#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : ImgDataGen.py
@Time    : 2021/1/7 15:24
@desc	 : 图片数据增广操作
'''

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# load_image
img = load_img('person1.png')

# img_to_array
x = img_to_array(img)

x = x.reshape((1,) + x.shape)

datagen = ImageDataGenerator(
    # 图片随机翻转的角度
    rotation_range=10,
    # 图片随机水平偏移的幅度
    width_shift_range=0.2,
    # 图片随机垂直偏移的幅度
    height_shift_range=0.2,
    # 执行其他处理前乘到整个图像上
    rescale=1. / 255,
    # 剪切强度
    shear_range=0.2,
    # 随机放大
    zoom_range=0.2,
    # 随机水平翻转cmd
    horizontal_flip=True,
    fill_mode='nearest')

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='per', save_format='jpg'):
    i += 1
    if i > 100:
        break
