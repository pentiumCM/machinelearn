#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : modify_file_suffix.py
@Time    : 2020/8/28 14:14
@desc	 : 修改文文件后缀
'''

# -*- coding: utf-8 -*-

import os


# # 列出当前目录下所有的文件
# filedir = 'C:\\Users\\WT\\Desktop\\test'
# files = os.listdir(filedir)   # 如果path为None，则使用 path = '.'
#
# for filename in files:
#     portion = os.path.splitext(filename)  # 分离文件名与扩展名
#     # 如果后缀是jpg
#     if portion[1] == '.jpg':
#         # 重新组合文件名与后缀名
#         newname = portion[0] + '.gif'
#         filename = filedir + '\\' + filename
#         newname = filedir + '\\' + newname
#         os.rename(filename, newname)


def replace_suffix(filedir, suffix):
    files = os.listdir(filedir)
    num = 0
    for filename in files:
        portion = os.path.splitext(filename)

        if portion[1] != suffix:
            newname = portion[0] + suffix
            filename = filedir + '\\' + filename
            newname = filedir + '\\' + newname
            os.rename(filename, newname)
            print("替换文件后缀", filename)
            num = num + 1
            print(num)


if __name__ == '__main__':
    replace_suffix('F:\\experiment\\data\\DC-ped-dataset_base\\validation\\non-ped_examples', '.jpg')
