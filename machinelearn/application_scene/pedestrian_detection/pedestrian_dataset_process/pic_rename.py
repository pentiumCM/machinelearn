#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : pic_rename.py
@Time    : 2020/12/4 23:35
@desc	 : 文件重命名
'''

import os

from utils.io_utils import get_subfiles

if __name__ == '__main__':

    file_dir1 = 'F:/develop_code/python/yolo/yolov5/code/kaist_ds/images/train/'
    file_dir2 = 'F:/develop_code/python/yolo/yolov5/code/kaist_ds/images/test/'

    files = get_subfiles(file_dir2)

    for file in files:
        file_names = file.split('_')

        new_file_name = file_names[0] + '_' + file_names[1] + '_' + file_names[2] + '.jpg'

        os.rename(file_dir2 + file, file_dir2 + new_file_name)
