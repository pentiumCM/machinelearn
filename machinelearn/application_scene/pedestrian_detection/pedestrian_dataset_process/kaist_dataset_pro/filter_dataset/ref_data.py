#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : ref_data.py
@Time    : 2020/11/19 20:59
@desc	 : 获取参考文献中的数据集
'''

import os
import shutil

from utils.io_utils import get_subfiles


def filter_ds(voc_labels_dir, image_dir, ref_labels_dir, filter_labels_dir, filter_image_dir):
    '''
    过滤数据集
    :param voc_labels_dir:  原始VOC的标签文件夹
    :param image_dir: 原始图片的文件夹
    :param ref_labels_dir: 参考过滤的标签文件夹
    :param filter_labels_dir: 生成过滤后的标签文件夹
    :param filter_image_dir: 过滤后图片的文件夹
    :return:
    '''

    ref_labels = get_subfiles(ref_labels_dir)

    if not os.path.exists(filter_labels_dir):
        os.makedirs(filter_labels_dir)

    if not os.path.exists(filter_image_dir):
        os.makedirs(filter_image_dir)

    count = 0
    for ref_label_item in ref_labels:
        ref_label_name = ref_label_item.split('.')[0]

        # voc类型的标签文件名
        voc_label_name = ref_label_name + '.xml'

        # 图片模态
        # image_modal_type = ['visible', 'lwir']
        image_modal_type = ['visible']

        voc_label_path = os.path.join(voc_labels_dir, voc_label_name).replace('\\', '/')
        voc_filter_label_path = os.path.join(filter_labels_dir, voc_label_name).replace('\\', '/')

        if os.path.exists(voc_label_path):

            for modal_type in image_modal_type:
                image_file_name = ref_label_name + '_' + modal_type + '.jpg'

                image_path = os.path.join(image_dir, image_file_name).replace('\\', '/')
                image_filter_path = os.path.join(filter_image_dir, image_file_name).replace('\\', '/')
                shutil.copyfile(image_path, image_filter_path)

            print('{}->{}'.format(voc_label_path, voc_filter_label_path))
            shutil.copyfile(voc_label_path, voc_filter_label_path)
        else:
            count = count + 1
            print(voc_label_path + '：文件不存在')
    print('未匹配文件数：{a}'.format(a=count))






if __name__ == '__main__':
    # 训练集1，测试集2

    voc_labels_dir = 'F:/experiment/data/KAIST/annotations/annotations-xml'
    image_dir = 'F:/experiment/data/KAIST/data_proc/JPEGImages'

    ref_labels_dir1 = 'F:/experiment/data/KAIST/annotations/ref/train_annotations/sanitized_annotations'
    ref_labels_dir2 = 'F:/experiment/data/KAIST/annotations/ref/test_annotations/annotations_KAIST_test_set'

    voc_filter_labels_dir1 = 'D:/VOC/lables/train'
    voc_filter_labels_dir2 = 'D:/VOC/lables/test'

    filter_image_dir1 = 'D:/VOC/images/train'
    filter_image_dir2 = 'D:/VOC/images/test'

    filter_ds(voc_labels_dir, image_dir, ref_labels_dir1, voc_filter_labels_dir1, filter_image_dir1)
