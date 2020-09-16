#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : use_model.py
@Time    : 2020/8/28 14:40
@desc	 : 
'''

from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array

import argparse
import numpy as np

from keras.models import load_model

model_path = 'first_model.h5'

img_width, img_height = 18, 36

img_path = 'test_img/person5.png'
img = load_img(img_path, grayscale=True, target_size=(img_width, img_height))
x = img_to_array(img)

# expand the dimension of the input tensor to 4
x = np.expand_dims(x, axis=0)

# 加载整个模型
model = load_model(model_path)

print('Generating class predictions for the input image...')
predicted_class = model.predict_classes(x)

print('Generating class probability predictions for the input image...')
predict_probabilities = model.predict_proba(x)

if predicted_class[0, 0] == 1:
    print('The input image is a pedestrian image.(包含行人)')
    print('With probability:', 100.0 * predict_probabilities[0, 0], '%')
else:
    print('The input image is a non-pedestrian image.（不包含行人）')
    print('With probability:', 100.0 * (1.0 - predict_probabilities[0, 0]), '%')
