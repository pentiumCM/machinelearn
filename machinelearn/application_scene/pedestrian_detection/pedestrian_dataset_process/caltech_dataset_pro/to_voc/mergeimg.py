#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : mergeimg.py
@Time    : 2020/9/16 16:21
@desc	 : JPEG文件分别放到两个统一的文件夹里
'''

import os
import shutil

if __name__ == "__main__":
    imgpath_in = 'F:/experiment/Caltech/VOC_process/JPEG'
    imgout = 'F:/experiment/Caltech/VOC_process/JPEGImages'
    for setsubdir in os.listdir(imgpath_in):
        print(setsubdir)
        if setsubdir[:3] == 'set':
            # set 目录
            setsubdir_path = os.path.join(imgpath_in, setsubdir).replace('\\', '/')

            for vsubdir in os.listdir(setsubdir_path):
                # V目录
                vsubdir_path = os.path.join(setsubdir_path, vsubdir).replace('\\', '/')

                for imgfile in os.listdir(vsubdir_path):
                    # img - in路径
                    imgfile_in = os.path.join(vsubdir_path, imgfile).replace('\\', '/')

                    # img - out路径
                    imgfile_outname = setsubdir + '_' + vsubdir + '_' + imgfile
                    imgfile_out = os.path.join(imgout, imgfile_outname).replace('\\', '/')

                    print('{}->{}'.format(imgfile_in, imgfile_out))
                    shutil.copyfile(imgfile_in, imgfile_out)
