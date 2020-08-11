#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : take_frame1.py
@Time    : 2020/8/6 11:23
@desc	 : 视频取帧
'''

import cv2

import uuid


def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    # numFrame = 0
    # numFrame = uuid.uuid4()
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # cv2.imshow('video', frame)
                # numFrame += 1
                numFrame = uuid.uuid4()
                newPath = svPath + str(numFrame) + ".jpg"
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        if cv2.waitKey(10) == 27:
            break


# getFrame('E:/tx/ch01_20200806105851.mp4', 'E:/tx/img2/')
getFrame('E:/tx/2020_0806_1615/ch01_20200806135540.mp4', 'E:/tx/img3/')
