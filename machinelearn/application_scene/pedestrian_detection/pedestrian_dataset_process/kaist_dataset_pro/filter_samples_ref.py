#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : filter_samples_ref.py
@Time    : 2021/2/4 22:12
@desc	 : 提取参考的训练的数据
'''
import shutil

import numpy as np

import os

label_dir_org = 'F:/experiment/dataset/KAIST/annotations/ref/train_annotations/sanitized_annotations'
xml_label_org = 'F:/experiment/dataset/KAIST/annotations/annotations-xml'

data_dir_org = 'F:/experiment/dataset/KAIST/data_proc/JPEGImages'


def get_subfiles(file_dir):
    '''
    获取某一文件夹下面的所有子文件
    :param file_dir:
    :return:
    '''
    subfiles_list = []
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录
        # print(file)  # 当前路径下所有非目录子文件
        subfiles_list = files
    return subfiles_list


def get_pic():
    dir_dist = 'D:/kaist/ref/train/data'

    label_dist = 'D:/kaist/ref'

    label_files = get_subfiles(label_dir_org)

    for sample in label_files:

        filename = sample.split('.')[0]

        file_rgbname = filename + '_visible' + '.jpg'
        file_lwirname = filename + '_lwir' + '.jpg'

        file_newname = filename + '.jpg'

        dir_dist_rgb = dir_dist + '/' + 'rgb'

        dir_dist_lwir = dir_dist + '/' + 'lwir'

        if not os.path.exists(dir_dist_rgb):
            os.makedirs(dir_dist_rgb)
        if not os.path.exists(dir_dist_lwir):
            os.makedirs(dir_dist_lwir)

        file_org_rgb = data_dir_org + '/' + file_rgbname
        file_org_lwir = data_dir_org + '/' + file_lwirname

        file_dist_rgb = os.path.join(dir_dist_rgb, file_newname)
        file_dist_lwir = os.path.join(dir_dist_lwir, file_newname)

        shutil.copyfile(file_org_rgb, file_dist_rgb)
        shutil.copyfile(file_org_lwir, file_dist_lwir)

        print('{}->{}'.format(file_org_rgb, file_dist_rgb))


def get_label():
    dir_dist = 'D:/kaist/ref/train/data'

    label_dist = 'D:/kaist/ref/train/label'

    label_files = get_subfiles(label_dir_org)

    for sample in label_files:

        filename = sample.split('.')[0]

        file_newname = filename + '.xml'

        dir_dist_label = label_dist

        if not os.path.exists(dir_dist_label):
            os.makedirs(dir_dist_label)

        file_org_label = xml_label_org + '/' + file_newname

        file_dist_label = os.path.join(dir_dist_label, file_newname)

        shutil.copyfile(file_org_label, file_dist_label)

        print('{}->{}'.format(file_org_label, file_dist_label))


if __name__ == '__main__':
    get_label()
