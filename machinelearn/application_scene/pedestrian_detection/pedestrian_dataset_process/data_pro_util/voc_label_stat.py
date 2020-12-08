#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : voc_label_stat.py
@Time    : 2020/12/4 23:56
@desc	 : VOC数据集标签类别汇总
'''

import xml.dom.minidom as xmldom
import os


def voc_label_statistics(annotation_path):
    '''
    voc数据集类别标签统计
    :param annotation_path: voc数据集的标签文件夹
    :return: [class1,class2.....]
    '''

    annotation_names = [os.path.join(annotation_path, i) for i in os.listdir(annotation_path)]

    labels = list()
    for names in annotation_names:

        print('process：', names)
        xmlfilepath = names
        domobj = xmldom.parse(xmlfilepath)
        # 得到元素对象
        elementobj = domobj.documentElement
        # 获得子标签
        subElementObj = elementobj.getElementsByTagName("object")
        for s in subElementObj:
            label = s.getElementsByTagName("name")[0].firstChild.data
            if label not in labels:
                labels.append(label)
    return labels


if __name__ == '__main__':
    annotation_path = "F:/experiment/data/KAIST/annotations/annotations-xml"
    # annotation_path = "F:/develop_code/python/yolo/yolov5/code/kaist_ds/lables/test"

    label = voc_label_statistics(annotation_path)

    print(label)