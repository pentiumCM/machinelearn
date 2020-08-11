#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : takeframe_by_second1.py
@Time    : 2020/8/6 18:06
@desc	 : 
'''

# 导入所需要的库
import cv2
import numpy as np

import uuid


# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)


def getFrame(videoPath, svPath):
    # 读取视频文件
    videoCapture = cv2.VideoCapture(videoPath)
    # 通过摄像头的方式
    # videoCapture=cv2.VideoCapture(1)

    # 读帧
    success, frame = videoCapture.read()
    i = 0
    timeF = 455

    while success:
        i = i + 1
        if (i % timeF == 0):
            # print('save image:', i)

            numFrame = uuid.uuid4()
            newPath = svPath + str(numFrame) + ".jpg"
            cv2.imencode('.jpg', frame)[1].tofile(newPath)

        success, frame = videoCapture.read()


getFrame('E:/tx/2020_0806_1615/ch01_20200806135540.mp4', 'E:/tx/pic3/')
