#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : borden_det.py
@Time    : 2021/4/27 19:26
@desc	 : 
'''

import cv2
import numpy as np

img = cv2.pyrDown(cv2.imread("circle.jpg", cv2.IMREAD_UNCHANGED))
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 创建新的图像black
black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)

for cnt in contours:
    # 轮廓周长也被称为弧长。可以使用函数 cv2.arcLength() 计算得到。这个函数的第二参数可以用来指定对象的形状是闭合的（True） ，还是打开的（一条曲线）
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    # 函数approxPolyDP来对指定的点集进行逼近，cnt是图像轮廓，epsilon表示的是精度，越小精度越高，因为表示的意思是是原始曲线与近似曲线之间的最大距离。
    # 第三个函数参数若为true,则说明近似曲线是闭合的，它的首位都是相连，反之，若为false，则断开。
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # convexHull检查一个曲线的凸性缺陷并进行修正，参数cnt是图像轮廓。
    hull = cv2.convexHull(cnt)
    # 勾画图像原始的轮廓
    cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    # 用多边形勾画轮廓区域
    cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
    # 修正凸性缺陷的轮廓区域
    cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)
# 显示图像
cv2.imshow("hull", black)
cv2.waitKey()
cv2.destroyAllWindows()
