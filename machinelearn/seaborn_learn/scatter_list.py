#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : scatter_list.py
@Time    : 2020/9/6 17:52
@desc	 : seaborn 可视化 list数据
'''

import seaborn as sns

import pandas as pd

wheel_parms = [
    {'left_wheel_center_x': 518.5, 'right_wheel_center_x': 518.5, 'wheel_last_coo': (527, 234), 'car_type': 'bus',
     'wheel_center_list': [518.5]},
    {'left_wheel_center_x': 519.0, 'right_wheel_center_x': 519.0, 'wheel_last_coo': (529, 234), 'car_type': 'bus',
     'wheel_center_list': [519.0]},
    {'left_wheel_center_x': 519.5, 'right_wheel_center_x': 519.5, 'wheel_last_coo': (528, 235), 'car_type': 'bus',
     'wheel_center_list': [519.5]},
    {'left_wheel_center_x': 515.5, 'right_wheel_center_x': 515.5, 'wheel_last_coo': (527, 236), 'car_type': 'bus',
     'wheel_center_list': [515.5]},
    {'left_wheel_center_x': 503.5, 'right_wheel_center_x': 503.5, 'wheel_last_coo': (522, 243), 'car_type': 'bus',
     'wheel_center_list': [503.5]},
    {'left_wheel_center_x': 469.0, 'right_wheel_center_x': 496.5, 'wheel_last_coo': (506, 261), 'car_type': 'bus',
     'wheel_center_list': [496.5]},
    {'left_wheel_center_x': 402.5, 'right_wheel_center_x': 452.5, 'wheel_last_coo': (472, 318), 'car_type': 'bus',
     'wheel_center_list': [402.5, 452.5]},
    {'left_wheel_center_x': 176.5, 'right_wheel_center_x': 346.5, 'wheel_last_coo': (395, 474), 'car_type': 'bus',
     'wheel_center_list': [176.5, 346.5]}
]

# 创建一个空的dataframe
wheel_data = pd.DataFrame(columns=["frame_index", "wheel_centers"])

for frame in range(len(wheel_parms)):
    wheel_center_list = wheel_parms[frame]['wheel_center_list']
    for item in range(len(wheel_center_list)):
        wheel_data.loc[wheel_data.shape[0]] = {"frame_index": frame, "wheel_centers": wheel_center_list[item]}

# 应用默认的默认seaborn主题，缩放比例和调色板
sns.set()

# Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
data = wheel_data

print(data.tail())

fig_name = 'wheel_list.png'

# fig_path为想要存入的文件夹或地址
fig = sns.scatterplot(x=data['frame_index'], y=data['wheel_centers'],
                      data=data, s=50)
scatter_fig = fig.get_figure()
scatter_fig.savefig(fig_name, dpi=400)
