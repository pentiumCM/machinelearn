#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : pca_utils.py
@Time    : 2020/4/29 21:12
@desc	 : 降维工具类
'''

from sklearn.decomposition import PCA


def pac_transfer_data(data, n_components):
    """
    数据降维处理
    :param data: 原始数据
    :param n_components: 指定保留主成分的个数
    :return: 返回降维后的数据
    """
    pca = PCA(n_components=n_components)
    pca.fit(data)
    pca_res = pca.transform(data)
    return pca_res
