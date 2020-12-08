#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : read_imgs.py
@Time    : 2020/2/14 11:37
@desc	 : opencv批量读取图片
'''

import os.path
import glob
import cv2


def convertjpg(jpgfile, outdir):
    src = cv2.imread(jpgfile, cv2.IMREAD_ANYCOLOR)

    try:
        cv2.imwrite(os.path.join(outdir, os.path.basename(jpgfile)), src)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    img = cv2.imread('F:/develop_code/python/machinelearn/machinelearn/cv_learn/visualization/cnn_5_img.jpg')

    print(img)

    cv2.imshow('img', img)
    cv2.waitKey(0)
