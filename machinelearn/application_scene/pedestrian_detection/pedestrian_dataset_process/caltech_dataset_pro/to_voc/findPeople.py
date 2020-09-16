#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : findPeople.py
@Time    : 2020/9/16 20:28
@desc	 : 将people标签替换成person
            Caltech的标注里有很多别的类别的行人，people，person
'''

# -*- coding:utf-8 -*-
import os
import re

if __name__ == "__main__":
    xmlin = 'F:/experiment/Caltech/VOC_process/Annotations'
    xmlout = 'F:/experiment/Caltech/VOC_process/Annotations_filter'
    files = os.listdir(xmlin)
    # 编译一个pattern
    pattern = re.compile('people')
    # 每张图片进行判断
    for file in files:
        f = open(os.path.join(xmlin, file), 'r')
        content = f.read()
        f.close()
        result = re.search('people', content)
        if (result != None):
            updateFile = pattern.sub('person', content)
        else:
            updateFile = content
        with open(os.path.join(xmlout, file), 'w') as fout:
            fout.write(updateFile)
        print('updating file {}'.format(file))
