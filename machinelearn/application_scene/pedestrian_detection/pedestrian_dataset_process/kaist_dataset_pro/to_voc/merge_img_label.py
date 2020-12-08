#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : merge_img_label.py
@Time    : 2020/9/16 16:21
@desc	 : 将数据集中的源图片和标签，分别放到对应的对应的文件中。JPEGImages，Annotations
'''

import os
import shutil


def merge_img(imgpath_in, imgpath_out):
    """
    将数据集图片存入一个文件夹中
    :param imgpath_in:
    :param imgpath_out:
    :return:
    """
    imgpath_in = imgpath_in
    imgpath_out = imgpath_out

    set_dirs = os.listdir(imgpath_in)
    for setsubdir in set_dirs:
        print(setsubdir)
        if setsubdir[:3] == 'set':
            # set 目录
            setsubdir_path = os.path.join(imgpath_in, setsubdir).replace('\\', '/')

            vdirs = os.listdir(setsubdir_path)
            for vsubdir in vdirs:
                # V目录
                vsubdir_path = os.path.join(setsubdir_path, vsubdir).replace('\\', '/')

                modaldirs = os.listdir(vsubdir_path)
                for modal in modaldirs:
                    # 模态文件夹：visible / lwir

                    modalimg_path = os.path.join(vsubdir_path, modal).replace('\\', '/')

                    images = os.listdir(modalimg_path)
                    for image in images:
                        img_file_name = image.split('.')[0]

                        # img - in路径
                        imgfile_in = os.path.join(modalimg_path, image).replace('\\', '/')
                        # img - out路径
                        imgfile_outname = setsubdir + '_' + vsubdir + '_' + img_file_name + '_' + modal + '.jpg'
                        imgfile_out = os.path.join(imgpath_out, imgfile_outname).replace('\\', '/')

                        print('{}->{}'.format(imgfile_in, imgfile_out))
                        shutil.copyfile(imgfile_in, imgfile_out)


def merge_label(labelpath_in, labelpath_out):
    """
    将数据集的标签存入一个文件夹中
    :param labelpath_in:
    :param labelpath_out:
    :return:
    """
    labelpath_in = labelpath_in
    labelpath_out = labelpath_out

    for labelfile in os.listdir(labelpath_in):
        if labelfile[:3] == 'set':
            labelfile_in = os.path.join(labelpath_in, labelfile).replace('\\', '/')

            labelfile_out = os.path.join(labelpath_out, labelfile)
            print('{}->{}'.format(labelfile_in, labelfile_out))
            shutil.copyfile(labelfile_in, labelfile_out)


if __name__ == "__main__":
    imgin = 'F:/experiment/data/KAIST/KAIST_DATA/dataset'
    imgout = 'F:/experiment/data/KAIST/data_proc/JPEGImages'
    merge_img(imgin, imgout)

    # labelin = 'F:/experiment/Caltech/VOC_process/annotations_in'
    # labelout = 'F:/experiment/Caltech/VOC_process/Annotations'
    # merge_label(labelin, labelout)
