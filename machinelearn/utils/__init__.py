#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/4/29 21:11
@desc	 : 项目的工具模块
'''

import pandas as pd

from utils.io_utils import get_subdirs


def gather_cam(file_path, position, excel_name):
    """
    整理摄像头的位置
    :param file_path: 录像路径
    :param position:  摄像头安装的位置
    :param excel_name: 导出的excel文件名
    :return:
    """
    dirs_list = get_subdirs(file_path)

    # 创建一个空的dataframe
    dir_df = pd.DataFrame(columns=["序号", "唯一编码", "桩号", "路面分割", "目标检测", "位置"])

    for item in range(len(dirs_list)):
        dir_str = dirs_list[item]
        str_main_split = dir_str.split()

        if (len(str_main_split) == 3):
            str_sub_split = str_main_split[2].split('_')
            # 桩号
            zh = str_main_split[0] + ' ' + str_main_split[1] + ' ' + str_sub_split[0]
        elif (len(str_main_split) == 2):
            str_sub_split = str_main_split[1].split('_')
            # 桩号
            zh = str_main_split[0] + ' ' + str_sub_split[0]

        # 唯一编码
        encode = str_sub_split[2]

        index = item + 1

        dir_df.loc[dir_df.shape[0]] = {"序号": index,
                                       "唯一编码": encode,
                                       "桩号": zh, "路面分割": '',
                                       "目标检测": '',
                                       "位置": position}

    print(dir_df.tail())

    dir_df.to_excel(excel_name + '.xls')


gather_cam('E:/data/y6/', '雍六', 'y6')
