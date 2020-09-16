#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : df2list.py
@Time    : 2020/9/11 15:39
@desc	 : dataFrame 转 List
'''

from utils.loaddata_utils import load_csvdata

df = load_csvdata('F:/project/tx/filter/wheel_2.csv')

print(df.tail())

frame_index_arr = df['frame_index'].values

length = int(frame_index_arr[frame_index_arr.shape[0] - 1])

wheel_parms = []

# 遍历每一帧
for i in range(length):
    wheel_center_list = []
    # 从每一帧角度遍历 df 每一行
    for j in range(0, len(df)):
        frame_index = int(df.iloc[j]['frame_index'])
        if (i < frame_index):
            break
        if (frame_index == i):
            wheel_center_list.append(df.iloc[i]['wheel_centers'])
    # df.iloc[i]['c1'], df.iloc[i]['c2']

    wheel_parms_item = {
        'frame_index': i,
        'wheel_center_list': wheel_center_list
    }
    wheel_parms.append(wheel_parms_item)

print(wheel_parms)
