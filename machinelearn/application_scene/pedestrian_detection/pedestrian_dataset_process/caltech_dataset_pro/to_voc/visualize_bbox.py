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


def visualize_bbox(xml_file, img_file):
    import cv2
    tree = etree.parse(xml_file)
    # load image
    image = cv2.imread(img_file)
    origin = cv2.imread(img_file)
    # 获取一张图片的所有bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        print(coord)
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imshow("test", image)
    cv2.imshow('origin', origin)
    cv2.waitKey(0)


if __name__ == "__main__":
    # main()

    # 可视化标注矩形框
    xml_file = "F:/experiment/Caltech/VOC_process/Annotations/080000.xml"
    img_file = "F:/experiment/Caltech/VOC_process/JPEGImages/080000.jpg"
    visualize_bbox(xml_file, img_file)
