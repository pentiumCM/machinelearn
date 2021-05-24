#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : pic_rename.py
@Time    : 2020/12/4 23:35
@desc	 : 文件重命名，去除模态标识
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


if __name__ == '__main__':

    file_dir1 = 'F:/project/jit/MrZ/examtestmis/code/yolov5-3.0/yolov5-exam/inference/images/'
    file_dir2 = 'F:/project/jit/MrZ/examtestmis/code/yolov5-3.0/yolov5-exam/inference/images/'

    files = get_subfiles(file_dir2)

    for file in files:
        # 未标题-1_0000_图层 455

        file_names = file

        file_name = file_names.replace('未标题', 'unname').replace('图层', 'layer')
        file_name = file_name.split(' ')
        new_file_name = file_name[0] + '_' + file_name[1]

        os.rename(file_dir2 + file, file_dir2 + new_file_name)
