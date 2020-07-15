#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : os_filter.py
@Time    : 2020/6/12 14:22
@desc	 : os过滤文件
'''

# -*- coding: utf-8 -*-

import os


def file_name(file_dir):
    '''
    遍历某个目录下面的所有子文件
    :param file_dir:目录名
    :return:
    '''
    file_list = []
    pro_files = []
    for files in os.walk(file_dir):
        file_list = files[2]
    for i in file_list:
        s = "".join(i)
        s = s.split('_')[0]
        pro_files.append(s)
    return pro_files


def file_del(path):
    '''
    删除某文件
    :param path:删除文件的路径
    :return:
    '''
    os.remove(path)


if __name__ == "__main__":

    pro_files = file_name('F:/samples/pic_org')

    test_files = file_name('F:/samples/pic_test')

    target_file = 'F:/samples/pic_org/'
    for i in test_files:
        file = i
        del_file = target_file + file

        # 判断文件是否存在，存在就删除
        if (os.path.exists(del_file)):
            print("删除文件：", del_file)
            file_del(del_file)
        else:
            print("要删除的文件不存在：", del_file)
