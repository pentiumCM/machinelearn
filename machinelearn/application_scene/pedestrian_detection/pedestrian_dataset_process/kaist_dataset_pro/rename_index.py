#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : rename_index.py
@Time    : 2020/9/16 17:20
@desc	 : 重命名图片和XML文件，
            重命名规则，根据已有的标注文件来匹配图片，命名以6位整数
'''

import os

xmlpath = 'F:/experiment/data/KAIST/data_proc/annotations-xml'
imgpath = 'F:/experiment/data/KAIST/data_proc/JPEGImages1'

index = 0
count = 0
emptyset = set()
xmlFiles = os.listdir(xmlpath)
imgFiles = os.listdir(imgpath)
print(len(xmlFiles), len(imgFiles))

for xml in xmlFiles:
    xmlname = os.path.splitext(xml)[0]
    imgname = os.path.join(imgpath, xmlname + '.jpg')
    if os.path.exists(imgname):
        newName = str(index).zfill(6)
        # 重命名图像
        os.rename(imgname, os.path.join(imgpath, newName + '.jpg'))
        # 重命名xml文件
        os.rename(os.path.join(xmlpath, xml), os.path.join(xmlpath, newName + '.xml'))
        print('============================================')
        print('img', imgname, os.path.join(imgpath, newName + '.jpg'))
        print('__________________________________________')
        print('xml', os.path.join(xmlpath, xml), os.path.join(xmlpath, newName + '.xml'))
        print('============================================')
        index = index + 1
    else:
        count += 1
        emptyset.add(xmlname.split('_')[0] + '_' + xmlname.split('_')[1])

sortedSet = sorted(emptyset, key=lambda x: (x.split('_')[0], x.split('_')[1]))
for i in sortedSet:
    print(i)
print(count)
