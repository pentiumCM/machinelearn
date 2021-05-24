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
import os

import cv2

import uuid


# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)


def getFrame(videoPath, svPath, timeF):
    """
    视频取帧
    :param videoPath: 视频源文件
    :param svPath: 帧保存路径
    :param timeF: 隔 timeF 帧取一帧，（查看视频帧率）
    :return:
    """
    # 读取视频文件
    videoCapture = cv2.VideoCapture(videoPath)
    # 通过摄像头的方式
    # videoCapture=cv2.VideoCapture(1)

    # 读帧
    success, frame = videoCapture.read()
    i = 0

    while success:
        i = i + 1
        if (i % timeF == 0):
            # print('save image:', i)

            numFrame = uuid.uuid4()
            newPath = svPath + str(numFrame) + ".jpg"
            cv2.imencode('.jpg', frame)[1].tofile(newPath)

        success, frame = videoCapture.read()


if __name__ == '__main__':
    timeF = 28

    src_dir = 'D:/system/Users/penti/Documents/WeChat Files/cm842679178/FileStorage/Video/2021-01/'
    src = 'D:/system/Users/penti/Documents/WeChat Files/cm842679178/FileStorage/Video/2021-01/0965d38015a9786825feb84bdac21b37.mp4'
    dist = 'E:/tx/gzl/'

    if not os.path.exists(dist):
        os.makedirs(dist)

    for root, dirs, files in os.walk(src_dir):
        print('')

    getFrame(
        'D:/system/Users/penti/Documents/WeChat Files/cm842679178/FileStorage/Video/2021-01/0965d38015a9786825feb84bdac21b37.mp4',
        'E:/tx/gzl/', timeF)
