#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : filtler.py
@Time    : 2020/9/7 18:13
@desc	 : 
'''

import pandas as pd

has_df = pd.read_excel(r'E:/data/公路发展集团/has2.xlsx')

print(has_df.tail())


zh_df = pd.read_excel(r'E:/data/公路发展集团/zh2.xls')


for r1 in range(zh_df.shape[0]):
    for r2 in range(has_df.shape[0]):
        # 只有data2的值不为空时才更新data1
        if (zh_df.iloc[r1]['唯一编码'] == has_df.iloc[r2]['唯一编码']):
            # 下面不能写为data1.iloc[r1]['b'],这样写数据不会更新
            # zh_df.iloc[r1, zh_df.columns == '路面分割'] = has_df.iloc[r2, has_df.columns == '路面分割']
            zh_df.at[r1, '目标检测'] = has_df.iloc[r2]['目标检测']
            break

print(zh_df.head())
zh_df.to_excel("test2.xls")
