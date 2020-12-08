#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : iou_utils.py
@Time    : 2020/11/22 15:32
@desc	 : iou的工具模块
'''

import numpy as np


# def union(au, bu, area_intersection):
#     area_a = (au[2] - au[0]) * (au[3] - au[1])
#     area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
#     area_union = area_a + area_b - area_intersection
#     return area_union
#
#
# def intersection(ai, bi):
#     x = max(ai[0], bi[0])
#     y = max(ai[1], bi[1])
#     w = min(ai[2], bi[2]) - x
#     h = min(ai[3], bi[3]) - y
#     if w < 0 or h < 0:
#         return 0
#     return w * h
#
#
# def iou(a, b):
#     if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
#         return 0.0
#
#     area_i = intersection(a, b)
#     area_u = union(a, b, area_i)
#
#     return float(area_i) / float(area_u + 1e-6)


def iou(box1, box2):
    """
    计算预测框和真实框之间的IoU
    :param box1:预测框
    :param box2: 真实框
    :return:
    """
    inter_upleft = np.maximum(box1[:2], box2[:2])
    inter_botright = np.minimum(box1[2:], box2[2:])

    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)

    # 交集面积
    inter = inter_wh[0] * inter_wh[1]

    # 真实框的面积
    area_gt = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # 预测框的面积
    area_pred = (box1[2] - box1[0]) * (box1[3] - box1[1])

    # 并集面积
    union = area_gt + area_pred - inter

    # 计算 IoU
    iou_val = inter / union
    return iou_val


if __name__ == '__main__':
    box1 = np.array([50, 50, 100, 100])
    box2 = np.array([200, 200, 300, 300])
    iou_val = iou(box1, box2)

    print(iou_val)
