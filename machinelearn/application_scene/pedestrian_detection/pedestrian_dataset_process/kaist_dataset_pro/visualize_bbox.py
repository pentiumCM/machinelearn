#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : visualize_bbox.py
@Time    : 2020/9/16 15:40
@desc	 : 可视化 bounding box
'''

from lxml import etree, objectify
import os
import cv2


def visualize_bbox(xml_file, img_file, img_name):
    """
    可视化矩形框
    :param xml_file: 标注文件
    :param img_file: 图片
    :param img_name: 图片名称
    :return:
    """
    tree = etree.parse(xml_file)

    # load image
    image = cv2.imread(img_file)
    # origin = cv2.imread(img_file)
    # 获取一张图片的所有bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        print(coord)
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imshow('label: ' + img_name, image)
    cv2.waitKey(0)


if __name__ == "__main__":

    # 模态类型：可见光 / 红外
    modal_type = ['visible', 'lwir']

    # 可视化标注矩形框
    img_file_dir = r"F:/experiment/data/KAIST/data_proc/JPEGImages"
    annotations_file_dir = r"F:/experiment/data/KAIST/annotations/annotations-xml"

    for parent, dirnames, filenames in os.walk(annotations_file_dir):
        for filename in filenames:
            img_file_name = filename.split('.')[0]

            for modal in modal_type:
                img_file = os.path.join(img_file_dir, img_file_name + '_' + modal + ".jpg").replace('\\', '/')
                xml_file = os.path.join(annotations_file_dir, filename).replace('\\', '/')
                visualize_bbox(xml_file, img_file, img_file_name + '_' + modal)

    # img_file = r"F:/experiment/data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000005.jpg"
    # xml_file = r"F:/experiment/data/VOC/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000005.xml"
