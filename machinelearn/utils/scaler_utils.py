#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : scaler_utils.py
@Time    : 2020/4/29 21:30
@desc	 : 数据标准化工具模块
'''

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def std_scaler_transdata(data):
    """
    数据标准化
    :param data: 原始数据
    :return:  返回标准化之后的数据
    """
    std_scaler = StandardScaler().fit(data)
    # 用scaler转换训练集
    transdata = std_scaler.transform(data)
    return transdata


def normalized_scaler_transdata(data):
    """
    数据归一化
    :param data: 原始数据
    :return:  返回归一化之后的数据
    """
    normalize_scaler = MinMaxScaler().fit(data)
    # 用scaler转换训练集
    transdata = normalize_scaler.transform(data)
    return transdata
