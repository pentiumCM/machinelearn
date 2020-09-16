#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : io_utils.py
@Time    : 2020/9/4 10:51
@desc	 : io工具类
'''

import os


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


def get_subdirs(file_dir):
    '''
    获取某一文件夹下面的所有子文件夹
    :param file_dir:
    :return:
    '''
    sub_dirs = []
    for root, dirs, files in os.walk(file_dir):
        for dir in dirs:
            # 获取目录的名称
            # print(dir)
            # 获取目录的路径
            # print(os.path.join(root, dir))
            sub_dirs.append(dir)
    return sub_dirs
