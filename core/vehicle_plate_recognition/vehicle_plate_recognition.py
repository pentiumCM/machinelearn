#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : vehicle_plate_recognition.py
@Time    : 2019/10/23 0:25
@desc	 : 
'''

import cv2
import numpy as np


# 形态学处理
def Process(img):
    # 高斯平滑
    gaussian = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    # Sobel算子
    # 梯度方向: x
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    # 核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    # 膨胀
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 膨胀
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    return dilation2


def GetRegion(img):
    regions = []
    # 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if (area < 2000):
            continue
        eps = 1e-3 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        if (ratio < 5 and ratio > 1.8):
            regions.append(box)
    return regions


def detect(img):
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prc = Process(gray)
    regions = GetRegion(prc)
    print('[INFO]:Detect %d license plates' % len(regions))
    for box in regions:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.imshow('Result', img)
    # 保存结果文件名
    cv2.imwrite('result2.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 输入的参数为图片的路径
    img = cv2.imread('F:/develop_code/python/dataset/vehicle_plate_recognition/test2.jpg')
    detect(img)
    detect(img)