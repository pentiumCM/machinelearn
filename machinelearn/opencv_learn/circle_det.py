#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : circle_det.py
@Time    : 2021/4/27 17:18
@desc	 : opencv —— 圆检测
'''

# 霍夫圆检测
import cv2 as cv
import numpy as np


def detect_circles_demo(image):
    # 灰度处理
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # medianBlur 平滑（模糊）处理
    img = cv.medianBlur(gray_img, 5)
    # 灰度图像转彩色图像
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    dst = cv.pyrMeanShiftFiltering(image, 10, 100)  # 边缘保留滤波EPF
    cimage = cv.cvtColor(dst, cv.COLOR_RGB2GRAY)

    # 圆检测
    # dp参数表示累加器与原始图像相比的分辨率的反比参数
    # minDist参数表示检测到的两个圆心之间的最小距离
    # param1参数表示Canny边缘检测的高阈值，低阈值会被自动置为高阈值的一半。
    # param2参数表示圆心检测的累加阈值，参数值越小，可以检测越多的假圆圈，但返回的是与较大累加器值对应的圆圈。
    # minRadius参数表示检测到的圆的最小半径。
    # maxRadius参数表示检测到的圆的最大半径。
    circles = cv.HoughCircles(image=cimage, method=cv.HOUGH_GRADIENT,
                              dp=1, minDist=10, param1=100, param2=30, minRadius=0, maxRadius=150)

    if circles is None:
        print("未检测到圆形！")
        return

        # 把circles包含的圆心和半径的值变成整数
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 勾画圆形，planets图像、(i[0],i[1])圆心坐标，i[2]是半径
        cv.circle(img=image, center=(i[0], i[1]), radius=i[2], color=(0, 0, 255), thickness=2)
        # 勾画圆心，圆心实质也是一个半径为2的圆形
        cv.circle(img=image, center=(i[0], i[1]), radius=2, color=(0, 0, 255), thickness=2)

    # 显示图像
    cv.imshow("circles", image)


if __name__ == '__main__':
    src = cv.imread('circle.jpg')
    cv.namedWindow('input_image', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
    cv.imshow('input_image', src)
    detect_circles_demo(src)
    cv.waitKey(0)
    cv.destroyAllWindows()
