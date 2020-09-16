#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : mergrxml.py
@Time    : 2020/9/16 16:34
@desc	 : 将所有的xml文件 以及 JPEG文件分别放到两个统一的文件夹里
'''

# -*- coding:utf-8 -*-
import os
import shutil

if __name__ == "__main__":
    xmlpathin = 'F:/experiment/Caltech/VOC_process/annotations_in'
    xmlout = 'F:/experiment/Caltech/VOC_process/Annotations'
    for xmlfile in os.listdir(xmlpathin):
        if xmlfile[:3] == 'set':
            xmlfile_in = os.path.join(xmlpathin, xmlfile).replace('\\', '/')

            xmlfile_out = os.path.join(xmlout, xmlfile)
            print('{}->{}'.format(xmlfile_in, xmlfile_out))
            shutil.copyfile(xmlfile_in, xmlfile_out)
