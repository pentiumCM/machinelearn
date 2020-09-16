#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : wheel_group.py
@Time    : 2020/9/10 23:59
@desc	 : 从检测的车轮中给车轮分组
'''

from application_scene.wheel_detection.wheel_parms import wheel_parms_6

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

wheel_parms = wheel_parms_6


def generate_left_wheel_fig(wheel_parms, fig_path, csv_path):
    """
    生成最左边车轮的坐标散点图
    :param data: 散点图的数据
    :param fig_path: 生成散点图的路径
    :return:
    """
    # 创建一个空的dataframe
    wheel_data = pd.DataFrame(columns=["frame_index", "wheel_group_x"])

    # 遍历车轮坐标列表，将数据放入 dataframe 中
    for frame in range(len(wheel_parms)):
        wheel_group_x = wheel_parms[frame]['wheel_group_x']
        frame_index = wheel_parms[frame]['frame_index']
        wheel_data.loc[wheel_data.shape[0]] = {"frame_index": frame_index, "wheel_group_x": wheel_group_x}

    # 应用默认的默认seaborn主题，缩放比例和调色板
    sns.set()
    # Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
    data = wheel_data
    fig = sns.scatterplot(x=data['frame_index'], y=data['wheel_group_x'],
                          data=data, s=50)

    plt.show()
    scatter_fig = fig.get_figure()
    # fig_path为想要存入的文件夹或地址
    scatter_fig.savefig('wheel_group_x' + fig_path, dpi=400)
    wheel_data.to_csv('wheel_group_x' + csv_path)


print(wheel_parms)

# 提取出的所有车轮轨迹
wheel_all_group = []

# 提取当前某一条的车轮轨迹
wheel_group = []

# 是否已全部遍历完,，初始状态为未遍历完
visited = False
while (bool(1 - visited)):
    visited = True
    for i in range(len(wheel_parms)):
        wheel_center_list = wheel_parms[i]['wheel_center_list']
        list.sort(wheel_center_list)
        if (len(wheel_center_list) > 0):
            # 从检测结果中取出当前帧的最左边的车轮坐标
            wheel_left_x = wheel_center_list[0]

            # 判断范围为 250 ~ 300 之间的点
            if (wheel_left_x >= 200 and wheel_left_x <= 300):
                wheel_group_item = {
                    'frame_index': i,
                    'wheel_group_x': wheel_left_x
                }
                # 如果当前条车轮轨迹为空,直接添加
                if (len(wheel_group) == 0):
                    wheel_group.append(wheel_group_item)
                    wheel_parms[i]['wheel_center_list'].remove(wheel_left_x)
                else:
                    # 取出当前条车轮轨迹的最后一个点
                    wheel_group_last_item = wheel_group[len(wheel_group) - 1]
                    wheel_group_last_x = wheel_group_last_item['wheel_group_x']
                    # 如果当前帧中最左车轮坐标比上一帧小,代表为同一车轮轨迹上
                    if (wheel_left_x <= wheel_group_last_x):
                        wheel_group.append(wheel_group_item)
                        wheel_parms[i]['wheel_center_list'].remove(wheel_left_x)
                    # 如果当前帧中最左车轮比上一帧大，代表这一车轮已经属于下一车轮，即本车轮轨迹已结束
                    else:
                        i = len(wheel_parms) - 1
            else:
                wheel_parms[i]['wheel_center_list'].remove(wheel_left_x)

        # 遍历完一个轨迹
        if (i == len(wheel_parms) - 1):
            wheel_all_group.append(wheel_group)
            wheel_group = []
        # 判断是否还有点没有遍历完
        if (len(wheel_parms[i]['wheel_center_list']) > 0):
            visited = False

print(wheel_all_group)

for i in range(len(wheel_all_group)):
    generate_left_wheel_fig(wheel_all_group[i], 'hello.png', 'hello.csv')

print('hello')
