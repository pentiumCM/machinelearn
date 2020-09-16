#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : seq2voc.py
@Time    : 2020/9/7 21:44
@desc	 : Caltech数据集 seq 类型文件数据集处理
'''

# Deal with .seq format for video sequence
# The .seq file is combined with images,
# so I split the file into several images with the image prefix
# "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46".

import os.path
import fnmatch
import shutil


def open_save(file, savepath):
    """
    read .seq file, and save the images into the savepath

    :param file: .seq文件路径
    :param savepath: 保存的图像路径
    :return:
    """

    # 读入一个seq文件，然后拆分成image存入savepath当中
    f = open(file, 'rb+')
    # 将seq文件的内容转化成str类型
    string = f.read().decode('latin-1')

    # splitstring是图片的前缀，可以理解成seq是以splitstring为分隔的多个jpg合成的文件
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

    # split函数做一个测试,因此返回结果的第一个是在seq文件中是空，因此后面省略掉第一个
    """
    >>> a = ".12121.3223.4343"
    >>> a.split('.')
    ['', '12121', '3223', '4343']
    """
    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
    # delete the image folder path if it exists
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    # create the image folder path
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # deal with file segment, every segment is an image except the first one
    for img in strlist:
        filename = str(count) + '.jpg'
        filenamewithpath = os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1


if __name__ == "__main__":
    rootdir = "F:/experiment/Caltech/data"
    saveroot = "F:/experiment/Caltech/VOC_process/JPEGImages"

    # walk in the rootdir, take down the .seq filename and filepath
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # check .seq file with suffix
            # fnmatch 全称是 filename match，主要是用来匹配文件名是否符合规则的
            if fnmatch.fnmatch(filename, '*.seq'):
                # take down the filename with path of .seq file
                thefilename = os.path.join(parent, filename)
                # create the image folder by combining .seq file path with .seq filename
                parent_path = parent
                parent_path = parent_path.replace('\\', '/')
                thesavepath = saveroot + '/' + parent_path.split('/')[-1] + '/' + filename.split('.')[0]
                print("Filename=" + thefilename)
                print("Savepath=" + thesavepath)
                open_save(thefilename, thesavepath)
