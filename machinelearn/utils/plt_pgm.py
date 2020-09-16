#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : plt_pgm.py
@Time    : 2020/8/28 15:41
@desc	 : 可视化pgm图像
'''

import os
from PIL import Image


def eachFile(filepath):
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if os.path.splitext(file)[1] == '.pgm':
                print(os.path.join(root, file))
                im = Image.open(os.path.join(root, file))
                im.show()
                print(im.size)


if __name__ == '__main__':
    filepath = "F:/experiment/data/DC-ped-dataset_base/ped_examples/"
    eachFile(filepath)
