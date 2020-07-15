#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : pytables_utils.py
@Time    : 2020/5/14 15:46
@desc	 : 
'''

import tables
import numpy as np

# 生成随机数据
data1 = np.random.random((1000, 4096))

hdf5_path = "test_data.hdf5"
# 和普通文件操作类似，'w'、'r'、'a' 分别表示写数据、读数据、追加数据
hdf5_file = tables.open_file(hdf5_path, mode='w')
# 设定压缩级别和压缩方法
filters = tables.Filters(complevel=5, complib='blosc')
earray = hdf5_file.create_earray(
    hdf5_file.root,
    'data',  # 数据名称，之后需要通过它来访问数据
    tables.Atom.from_dtype(data1.dtype),  # 设定数据格式（和data1格式相同）
    shape=(0, 4096),  # 第一维的 0 表示数据可沿行扩展
    filters=filters,
    expectedrows=1000000  # 完整数据大约规模，可以帮助程序提高时空利用效率
)
# 将 data1 添加进 earray
earray.append(data1)
# 写完之后记得关闭文件
hdf5_file.close()
