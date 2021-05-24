#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : rectangle.py
@Time    : 2021/3/5 15:26
@desc	 : 
'''

# coding=utf-8
# 导入python包
import numpy as np
import argparse
import imutils
import cv2

image_path = 'F:/develop_code/python/machinelearn/machinelearn/cv_learn/peopleincar5.jpg'

# 读取图片并将其转化为灰度图片
image = cv2.imread(image_path)
image1 = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算图像中x和y方向的Scharr梯度幅值表示
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

# x方向的梯度减去y方向的梯度
gradient = cv2.subtract(gradX, gradY)
# 获取处理后的绝对值
gradient = cv2.convertScaleAbs(gradient)
cv2.imwrite("gradient.png", gradient)

# 对处理后的结果进行模糊操作
blurred = cv2.blur(gradient, (9, 9))
# 将其转化为二值图片
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imwrite("thresh.png", thresh)

# 构建一个掩码并将其应用在二值图片中
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closed1.png", closed)

# 执行多次膨胀和腐蚀操作
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

cv2.imwrite("closed2.png", closed)

# 在二值图像中寻找轮廓, 然后根据他们的区域大小对该轮廓进行排序，保留最大的一个轮廓
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# 旋转图像
################################################################
rect = cv2.minAreaRect(c)
(h, w) = closed.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, rect[2], 1.0)
rotated = cv2.warpAffine(closed, M, (w, h))
cv2.imshow("Rotated", rotated)

#################################################################

# 计算最大的轮廓的最小外接矩形
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)

# 绘制并显示结果
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.drawContours(image1, [box], -1, (0, 255, 0), 3)
result = np.hstack([image, image1])
cv2.imwrite("detect1.png", result)
cv2.imshow("Image", result)
cv2.waitKey(0)
