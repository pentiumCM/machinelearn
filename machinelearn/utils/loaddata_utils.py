#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : loaddata_utils.py
@Time    : 2020/5/3 18:02
@desc	 : 加载数据集的工具模块
'''

import pandas as pd


def load_csvdata(datafile_path):
    """
    读取CSV文件的数据
    :param datafile_path: CSV文件的路径
    :return: 返回CSV文件的数据
    """
    datafile = datafile_path
    data = pd.read_csv(datafile)
    return data


def load_exceldata(datafile_path, sheet_name, index_col):
    data = pd.read_excel(io=datafile_path, sheet_name=sheet_name, index_col=index_col)
    return data
