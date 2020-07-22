#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : anchor_iml.py
@Time    : 2020/7/21 18:32
@desc	 : 锚框的实现
'''

import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# feature map的尺寸
size_Y = 16
size_X = 16

# 原始图片到feature map的缩放比例。类似卷积
rpn_stride = 8  # image_size = 16 * 8

# 锚框的尺度
scales = [1, 2, 4]
# 锚框的形状
ratios = [0.5, 1, 2]


def anchor_gen(size_X, size_Y, rpn_stride, scales, ratios):
    """
    构造锚框
    :param size_X:
    :param size_Y:
    :param rpn_stride:
    :param scales:
    :param ratios:
    :return:
    """

    # np.meshgrid：从坐标向量返回坐标矩阵
    # 把尺寸和比例组合一下
    scales, ratios = np.meshgrid(scales, ratios)
    scales, ratios = scales.flatten(), ratios.flatten()

    # 锚框的宽和高
    scalesY = scales * np.sqrt(ratios)
    scalesX = scales / np.sqrt(ratios)

    # 锚点（原始图片上的坐标点） = feature map坐标点 * rpn_stride
    shiftX = np.arange(0, size_X) * rpn_stride
    shiftY = np.arange(0, size_Y) * rpn_stride
    shiftX, shiftY = np.meshgrid(shiftX, shiftY)

    # 有16个锚点，每个锚点上面有9个锚框
    # centerX：锚框中心点的坐标
    centerX, anchorX = np.meshgrid(shiftX, scalesX)
    centerY, anchorY = np.meshgrid(shiftY, scalesY)

    anchor_center = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    anchor_size = np.stack([anchorY, anchorX], axis=2).reshape(-1, 2)

    # 锚框：左上顶点坐标 + 右下顶点坐标
    boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=1)
    return boxes


anchors = anchor_gen(size_X, size_Y, rpn_stride, scales, ratios)
print(anchors.shape)

plt.figure(figsize=(10, 10))

img = np.ones((128, 128, 3))

plt.imshow(img)

# get current Axs
Axs = plt.gca()

for i in range(anchors.shape[0]):
    box = anchors[i]
    rec = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], edgecolor='r', facecolor='none')
    Axs.add_patch(rec)

plt.show()
