#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : scatter_utils.py
@Time    : 2020/9/8 9:05
@desc	 : 
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wheel_parms = [
    {'left_wheel_center_x': 450.0, 'right_wheel_center_x': 484.0, 'wheel_last_coo': (516, 278), 'car_type': 'truck',
     'wheel_center_list': [450.0, 484.0]},
    {'left_wheel_center_x': 448.5, 'right_wheel_center_x': 448.5, 'wheel_last_coo': (518, 280), 'car_type': 'truck',
     'wheel_center_list': [448.5]},
    {'left_wheel_center_x': 448.0, 'right_wheel_center_x': 482.0, 'wheel_last_coo': (519, 280), 'car_type': 'truck',
     'wheel_center_list': [448.0, 482.0]},
    {'left_wheel_center_x': 447.0, 'right_wheel_center_x': 481.5, 'wheel_last_coo': (519, 280), 'car_type': 'truck',
     'wheel_center_list': [481.5, 447.0]},
    {'left_wheel_center_x': 446.0, 'right_wheel_center_x': 481.5, 'wheel_last_coo': (519, 281), 'car_type': 'truck',
     'wheel_center_list': [481.5, 446.0]},
    {'left_wheel_center_x': 444.5, 'right_wheel_center_x': 479.0, 'wheel_last_coo': (516, 283), 'car_type': 'truck',
     'wheel_center_list': [444.5, 479.0]},
    {'left_wheel_center_x': 442.0, 'right_wheel_center_x': 481.0, 'wheel_last_coo': (516, 283), 'car_type': 'truck',
     'wheel_center_list': [481.0, 442.0]},
    {'left_wheel_center_x': 442.0, 'right_wheel_center_x': 478.5, 'wheel_last_coo': (515, 283), 'car_type': 'truck',
     'wheel_center_list': [442.0, 478.5]},
    {'left_wheel_center_x': 441.0, 'right_wheel_center_x': 478.5, 'wheel_last_coo': (513, 283), 'car_type': 'truck',
     'wheel_center_list': [441.0, 478.5]},
    {'left_wheel_center_x': 440.5, 'right_wheel_center_x': 477.5, 'wheel_last_coo': (514, 290), 'car_type': 'truck',
     'wheel_center_list': [440.5, 477.5]},
    {'left_wheel_center_x': 439.5, 'right_wheel_center_x': 477.5, 'wheel_last_coo': (507, 293), 'car_type': 'truck',
     'wheel_center_list': [477.5, 439.5]},
    {'left_wheel_center_x': 438.5, 'right_wheel_center_x': 477.0, 'wheel_last_coo': (507, 294), 'car_type': 'truck',
     'wheel_center_list': [477.0, 438.5]},
    {'left_wheel_center_x': 435.5, 'right_wheel_center_x': 475.0, 'wheel_last_coo': (510, 294), 'car_type': 'truck',
     'wheel_center_list': [435.5, 475.0]},
    {'left_wheel_center_x': 434.5, 'right_wheel_center_x': 475.0, 'wheel_last_coo': (511, 294), 'car_type': 'truck',
     'wheel_center_list': [434.5, 475.0]},
    {'left_wheel_center_x': 432.5, 'right_wheel_center_x': 474.5, 'wheel_last_coo': (502, 297), 'car_type': 'truck',
     'wheel_center_list': [432.5, 474.5]},
    {'left_wheel_center_x': 432.0, 'right_wheel_center_x': 473.5, 'wheel_last_coo': (503, 299), 'car_type': 'truck',
     'wheel_center_list': [432.0, 473.5]},
    {'left_wheel_center_x': 430.0, 'right_wheel_center_x': 473.0, 'wheel_last_coo': (503, 301), 'car_type': 'truck',
     'wheel_center_list': [473.0, 430.0]},
    {'left_wheel_center_x': 428.5, 'right_wheel_center_x': 471.5, 'wheel_last_coo': (502, 302), 'car_type': 'truck',
     'wheel_center_list': [428.5, 471.5]},
    {'left_wheel_center_x': 428.0, 'right_wheel_center_x': 471.5, 'wheel_last_coo': (505, 307), 'car_type': 'truck',
     'wheel_center_list': [428.0, 471.5]},
    {'left_wheel_center_x': 427.0, 'right_wheel_center_x': 471.5, 'wheel_last_coo': (505, 308), 'car_type': 'truck',
     'wheel_center_list': [427.0, 471.5]},
    {'left_wheel_center_x': 425.5, 'right_wheel_center_x': 471.5, 'wheel_last_coo': (505, 308), 'car_type': 'truck',
     'wheel_center_list': [425.5, 471.5]},
    {'left_wheel_center_x': 423.5, 'right_wheel_center_x': 470.0, 'wheel_last_coo': (503, 307), 'car_type': 'truck',
     'wheel_center_list': [470.0, 423.5]},
    {'left_wheel_center_x': 422.0, 'right_wheel_center_x': 470.0, 'wheel_last_coo': (502, 311), 'car_type': 'truck',
     'wheel_center_list': [422.0, 470.0]},
    {'left_wheel_center_x': 420.5, 'right_wheel_center_x': 469.5, 'wheel_last_coo': (503, 312), 'car_type': 'truck',
     'wheel_center_list': [420.5, 469.5]},
    {'left_wheel_center_x': 417.5, 'right_wheel_center_x': 469.0, 'wheel_last_coo': (501, 314), 'car_type': 'truck',
     'wheel_center_list': [417.5, 469.0]},
    {'left_wheel_center_x': 416.5, 'right_wheel_center_x': 466.0, 'wheel_last_coo': (501, 316), 'car_type': 'truck',
     'wheel_center_list': [416.5, 466.0]},
    {'left_wheel_center_x': 415.5, 'right_wheel_center_x': 466.0, 'wheel_last_coo': (500, 316), 'car_type': 'truck',
     'wheel_center_list': [466.0, 415.5]},
    {'left_wheel_center_x': 413.5, 'right_wheel_center_x': 466.0, 'wheel_last_coo': (500, 317), 'car_type': 'truck',
     'wheel_center_list': [466.0, 413.5]},
    {'left_wheel_center_x': 413.0, 'right_wheel_center_x': 465.0, 'wheel_last_coo': (500, 319), 'car_type': 'truck',
     'wheel_center_list': [465.0, 413.0]},
    {'left_wheel_center_x': 410.0, 'right_wheel_center_x': 464.0, 'wheel_last_coo': (494, 322), 'car_type': 'truck',
     'wheel_center_list': [464.0, 410.0]},
    {'left_wheel_center_x': 408.5, 'right_wheel_center_x': 463.0, 'wheel_last_coo': (494, 323), 'car_type': 'truck',
     'wheel_center_list': [408.5, 463.0]},
    {'left_wheel_center_x': 405.0, 'right_wheel_center_x': 462.5, 'wheel_last_coo': (494, 326), 'car_type': 'truck',
     'wheel_center_list': [405.0, 462.5]},
    {'left_wheel_center_x': 404.5, 'right_wheel_center_x': 462.5, 'wheel_last_coo': (493, 328), 'car_type': 'truck',
     'wheel_center_list': [404.5, 462.5]},
    {'left_wheel_center_x': 403.0, 'right_wheel_center_x': 462.0, 'wheel_last_coo': (494, 333), 'car_type': 'truck',
     'wheel_center_list': [403.0, 462.0]},
    {'left_wheel_center_x': 400.0, 'right_wheel_center_x': 462.0, 'wheel_last_coo': (492, 336), 'car_type': 'truck',
     'wheel_center_list': [400.0, 462.0]},
    {'left_wheel_center_x': 392.5, 'right_wheel_center_x': 459.0, 'wheel_last_coo': (497, 341), 'car_type': 'truck',
     'wheel_center_list': [392.5, 459.0]},
    {'left_wheel_center_x': 391.5, 'right_wheel_center_x': 458.0, 'wheel_last_coo': (497, 341), 'car_type': 'truck',
     'wheel_center_list': [458.0, 391.5]},
    {'left_wheel_center_x': 388.0, 'right_wheel_center_x': 456.5, 'wheel_last_coo': (497, 343), 'car_type': 'truck',
     'wheel_center_list': [456.5, 388.0]},
    {'left_wheel_center_x': 387.0, 'right_wheel_center_x': 455.5, 'wheel_last_coo': (497, 344), 'car_type': 'truck',
     'wheel_center_list': [387.0, 455.5]},
    {'left_wheel_center_x': 382.5, 'right_wheel_center_x': 454.0, 'wheel_last_coo': (498, 349), 'car_type': 'truck',
     'wheel_center_list': [382.5, 454.0]},
    {'left_wheel_center_x': 380.5, 'right_wheel_center_x': 453.5, 'wheel_last_coo': (498, 353), 'car_type': 'truck',
     'wheel_center_list': [380.5, 453.5]},
    {'left_wheel_center_x': 377.5, 'right_wheel_center_x': 452.5, 'wheel_last_coo': (498, 354), 'car_type': 'truck',
     'wheel_center_list': [377.5, 452.5]},
    {'left_wheel_center_x': 372.0, 'right_wheel_center_x': 451.5, 'wheel_last_coo': (495, 362), 'car_type': 'truck',
     'wheel_center_list': [451.5, 372.0]},
    {'left_wheel_center_x': 371.0, 'right_wheel_center_x': 450.0, 'wheel_last_coo': (494, 365), 'car_type': 'truck',
     'wheel_center_list': [450.0, 371.0]},
    {'left_wheel_center_x': 364.5, 'right_wheel_center_x': 449.0, 'wheel_last_coo': (497, 371), 'car_type': 'truck',
     'wheel_center_list': [449.0, 364.5]},
    {'left_wheel_center_x': 361.5, 'right_wheel_center_x': 447.5, 'wheel_last_coo': (496, 375), 'car_type': 'truck',
     'wheel_center_list': [361.5, 447.5]},
    {'left_wheel_center_x': 356.5, 'right_wheel_center_x': 445.0, 'wheel_last_coo': (496, 382), 'car_type': 'truck',
     'wheel_center_list': [356.5, 445.0]},
    {'left_wheel_center_x': 354.0, 'right_wheel_center_x': 444.5, 'wheel_last_coo': (494, 385), 'car_type': 'truck',
     'wheel_center_list': [354.0, 444.5]},
    {'left_wheel_center_x': 342.5, 'right_wheel_center_x': 439.0, 'wheel_last_coo': (491, 394), 'car_type': 'truck',
     'wheel_center_list': [342.5, 439.0]},
    {'left_wheel_center_x': 337.5, 'right_wheel_center_x': 437.5, 'wheel_last_coo': (491, 399), 'car_type': 'truck',
     'wheel_center_list': [337.5, 437.5]},
    {'left_wheel_center_x': 331.0, 'right_wheel_center_x': 436.5, 'wheel_last_coo': (488, 409), 'car_type': 'truck',
     'wheel_center_list': [331.0, 436.5]},
    {'left_wheel_center_x': 326.0, 'right_wheel_center_x': 436.5, 'wheel_last_coo': (487, 416), 'car_type': 'truck',
     'wheel_center_list': [436.5, 326.0]},
    {'left_wheel_center_x': 321.0, 'right_wheel_center_x': 435.0, 'wheel_last_coo': (487, 419), 'car_type': 'truck',
     'wheel_center_list': [321.0, 435.0]},
    {'left_wheel_center_x': 310.5, 'right_wheel_center_x': 435.5, 'wheel_last_coo': (487, 431), 'car_type': 'truck',
     'wheel_center_list': [435.5, 310.5]},
    {'left_wheel_center_x': 305.0, 'right_wheel_center_x': 433.0, 'wheel_last_coo': (486, 437), 'car_type': 'truck',
     'wheel_center_list': [305.0, 433.0]},
    {'left_wheel_center_x': 292.5, 'right_wheel_center_x': 431.5, 'wheel_last_coo': (479, 447), 'car_type': 'truck',
     'wheel_center_list': [292.5, 431.5]},
    {'left_wheel_center_x': 287.0, 'right_wheel_center_x': 431.0, 'wheel_last_coo': (478, 456), 'car_type': 'truck',
     'wheel_center_list': [287.0, 431.0]},
    {'left_wheel_center_x': 272.5, 'right_wheel_center_x': 427.5, 'wheel_last_coo': (478, 476), 'car_type': 'truck',
     'wheel_center_list': [272.5, 427.5]},
    {'left_wheel_center_x': 264.5, 'right_wheel_center_x': 425.5, 'wheel_last_coo': (478, 480), 'car_type': 'truck',
     'wheel_center_list': [264.5, 425.5]},
    {'left_wheel_center_x': 248.0, 'right_wheel_center_x': 422.5, 'wheel_last_coo': (477, 481), 'car_type': 'truck',
     'wheel_center_list': [248.0, 422.5]},
    {'left_wheel_center_x': 236.5, 'right_wheel_center_x': 420.0, 'wheel_last_coo': (475, 477), 'car_type': 'truck',
     'wheel_center_list': [420.0, 236.5]},
    {'left_wheel_center_x': 213.0, 'right_wheel_center_x': 416.5, 'wheel_last_coo': (472, 479), 'car_type': 'truck',
     'wheel_center_list': [416.5, 213.0]},
    {'left_wheel_center_x': 200.5, 'right_wheel_center_x': 415.0, 'wheel_last_coo': (472, 479), 'car_type': 'truck',
     'wheel_center_list': [200.5, 415.0]},
    {'left_wheel_center_x': 191.0, 'right_wheel_center_x': 413.5, 'wheel_last_coo': (472, 479), 'car_type': 'truck',
     'wheel_center_list': [413.5, 191.0]},
    {'left_wheel_center_x': 167.5, 'right_wheel_center_x': 410.0, 'wheel_last_coo': (471, 480), 'car_type': 'truck',
     'wheel_center_list': [410.0, 167.5]},
    {'left_wheel_center_x': 155.0, 'right_wheel_center_x': 407.5, 'wheel_last_coo': (471, 478), 'car_type': 'truck',
     'wheel_center_list': [155.0, 407.5]},
    {'left_wheel_center_x': 126.0, 'right_wheel_center_x': 402.0, 'wheel_last_coo': (469, 475), 'car_type': 'truck',
     'wheel_center_list': [402.0, 126.0]},
    {'left_wheel_center_x': 112.0, 'right_wheel_center_x': 397.5, 'wheel_last_coo': (470, 477), 'car_type': 'truck',
     'wheel_center_list': [112.0, 397.5]},
    {'left_wheel_center_x': 95.0, 'right_wheel_center_x': 397.5, 'wheel_last_coo': (470, 475), 'car_type': 'truck',
     'wheel_center_list': [397.5, 95.0]},
    {'left_wheel_center_x': 396.0, 'right_wheel_center_x': 396.0, 'wheel_last_coo': (473, 476), 'car_type': 'truck',
     'wheel_center_list': [396.0]},
    {'left_wheel_center_x': 391.0, 'right_wheel_center_x': 391.0, 'wheel_last_coo': (473, 475), 'car_type': 'truck',
     'wheel_center_list': [391.0]},
    {'left_wheel_center_x': 45.5, 'right_wheel_center_x': 382.5, 'wheel_last_coo': (470, 475), 'car_type': 'truck',
     'wheel_center_list': [45.5, 382.5]},
    {'left_wheel_center_x': 382.0, 'right_wheel_center_x': 382.0, 'wheel_last_coo': (468, 476), 'car_type': 'truck',
     'wheel_center_list': [382.0]},
    {'left_wheel_center_x': 377.5, 'right_wheel_center_x': 377.5, 'wheel_last_coo': (465, 474), 'car_type': 'truck',
     'wheel_center_list': [377.5]},
    {'left_wheel_center_x': 375.0, 'right_wheel_center_x': 375.0, 'wheel_last_coo': (465, 474), 'car_type': 'truck',
     'wheel_center_list': [375.0]},
    {'left_wheel_center_x': 366.0, 'right_wheel_center_x': 366.0, 'wheel_last_coo': (464, 475), 'car_type': 'truck',
     'wheel_center_list': [366.0]},
    {'left_wheel_center_x': 361.5, 'right_wheel_center_x': 361.5, 'wheel_last_coo': (463, 475), 'car_type': 'truck',
     'wheel_center_list': [361.5]},
    {'left_wheel_center_x': 362.5, 'right_wheel_center_x': 362.5, 'wheel_last_coo': (460, 474), 'car_type': 'truck',
     'wheel_center_list': [362.5]},
    {'left_wheel_center_x': 353.5, 'right_wheel_center_x': 353.5, 'wheel_last_coo': (458, 472), 'car_type': 'truck',
     'wheel_center_list': [353.5]},
    {'left_wheel_center_x': 350.0, 'right_wheel_center_x': 350.0, 'wheel_last_coo': (457, 473), 'car_type': 'truck',
     'wheel_center_list': [350.0]},
    {'left_wheel_center_x': 338.5, 'right_wheel_center_x': 338.5, 'wheel_last_coo': (451, 473), 'car_type': 'truck',
     'wheel_center_list': [338.5]},
    {'left_wheel_center_x': 336.0, 'right_wheel_center_x': 336.0, 'wheel_last_coo': (446, 474), 'car_type': 'truck',
     'wheel_center_list': [336.0]},
    {'left_wheel_center_x': 328.0, 'right_wheel_center_x': 328.0, 'wheel_last_coo': (441, 471), 'car_type': 'truck',
     'wheel_center_list': [328.0]},
    {'left_wheel_center_x': 316.0, 'right_wheel_center_x': 316.0, 'wheel_last_coo': (438, 476), 'car_type': 'truck',
     'wheel_center_list': [316.0]},
    {'left_wheel_center_x': 307.0, 'right_wheel_center_x': 307.0, 'wheel_last_coo': (435, 476), 'car_type': 'truck',
     'wheel_center_list': [307.0]},
    {'left_wheel_center_x': 304.0, 'right_wheel_center_x': 304.0, 'wheel_last_coo': (430, 474), 'car_type': 'truck',
     'wheel_center_list': [304.0]},
    {'left_wheel_center_x': 300.0, 'right_wheel_center_x': 300.0, 'wheel_last_coo': (427, 477), 'car_type': 'truck',
     'wheel_center_list': [300.0]},
    {'left_wheel_center_x': 282.5, 'right_wheel_center_x': 282.5, 'wheel_last_coo': (420, 477), 'car_type': 'truck',
     'wheel_center_list': [282.5]},
    {'left_wheel_center_x': 269.0, 'right_wheel_center_x': 269.0, 'wheel_last_coo': (418, 475), 'car_type': 'truck',
     'wheel_center_list': [269.0]},
    {'left_wheel_center_x': 251.5, 'right_wheel_center_x': 251.5, 'wheel_last_coo': (407, 477), 'car_type': 'truck',
     'wheel_center_list': [251.5]},
    {'left_wheel_center_x': 237.5, 'right_wheel_center_x': 237.5, 'wheel_last_coo': (402, 478), 'car_type': 'truck',
     'wheel_center_list': [237.5]},
    {'left_wheel_center_x': 211.5, 'right_wheel_center_x': 211.5, 'wheel_last_coo': (395, 477), 'car_type': 'truck',
     'wheel_center_list': [211.5]},
    {'left_wheel_center_x': 190.0, 'right_wheel_center_x': 190.0, 'wheel_last_coo': (394, 478), 'car_type': 'truck',
     'wheel_center_list': [190.0]},
    {'left_wheel_center_x': 185.0, 'right_wheel_center_x': 185.0, 'wheel_last_coo': (392, 480), 'car_type': 'truck',
     'wheel_center_list': [185.0]},
    {'left_wheel_center_x': 153.0, 'right_wheel_center_x': 153.0, 'wheel_last_coo': (383, 474), 'car_type': 'truck',
     'wheel_center_list': [153.0]},
    {'left_wheel_center_x': 131.5, 'right_wheel_center_x': 131.5, 'wheel_last_coo': (376, 477), 'car_type': 'truck',
     'wheel_center_list': [131.5]},
    {'left_wheel_center_x': 102.5, 'right_wheel_center_x': 102.5, 'wheel_last_coo': (363, 477), 'car_type': 'truck',
     'wheel_center_list': [102.5]}]


def generate_all_wheel_fig(wheel_parms, fig_path, csv_path):
    """
    生成车轮的坐标散点图
    :param data: 散点图的数据
    :param fig_path: 生成散点图的路径
    :return:
    """
    # wheel_parms = [
    #     {'left_wheel_center_x': 518.5, 'right_wheel_center_x': 518.5, 'wheel_last_coo': (527, 234), 'car_type': 'bus',
    #      'wheel_center_list': [518.5]},
    #     {'left_wheel_center_x': 519.0, 'right_wheel_center_x': 519.0, 'wheel_last_coo': (529, 234), 'car_type': 'bus',
    #      'wheel_center_list': [519.0]}]

    # 创建一个空的dataframe
    wheel_data = pd.DataFrame(columns=["frame_index", "wheel_centers"])

    # 遍历车轮坐标列表，将数据放入 dataframe 中
    for frame in range(len(wheel_parms)):
        wheel_center_list = wheel_parms[frame]['wheel_center_list']
        wheel_center_list.sort()

        for item in range(len(wheel_center_list)):
            wheel_data.loc[wheel_data.shape[0]] = {"frame_index": frame, "wheel_centers": wheel_center_list[item]}

    # 应用默认的默认seaborn主题，缩放比例和调色板
    sns.set()
    # Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
    data = wheel_data
    fig = sns.scatterplot(x=data['frame_index'], y=data['wheel_centers'],
                          data=data, s=100)
    scatter_fig = fig.get_figure()
    # fig_path为想要存入的文件夹或地址
    scatter_fig.savefig('all_wheel' + fig_path, dpi=400)
    wheel_data.to_csv('all_wheel' + csv_path)
    plt.clf()


def generate_left_wheel_fig(wheel_parms, fig_path, csv_path):
    """
    生成最左边车轮的坐标散点图
    :param data: 散点图的数据
    :param fig_path: 生成散点图的路径
    :return:
    """
    # 创建一个空的dataframe
    wheel_data = pd.DataFrame(columns=["frame_index", "left_wheel_center_x"])

    # 遍历车轮坐标列表，将数据放入 dataframe 中
    for frame in range(len(wheel_parms)):
        left_wheel_center_x = wheel_parms[frame]['left_wheel_center_x']
        wheel_data.loc[wheel_data.shape[0]] = {"frame_index": frame, "left_wheel_center_x": left_wheel_center_x}

    # 应用默认的默认seaborn主题，缩放比例和调色板
    sns.set()
    # Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
    data = wheel_data
    fig = sns.scatterplot(x=data['frame_index'], y=data['left_wheel_center_x'],
                          data=data, s=50)
    scatter_fig = fig.get_figure()
    # fig_path为想要存入的文件夹或地址
    scatter_fig.savefig('left_wheel_center_x_' + fig_path, dpi=400)
    wheel_data.to_csv('left_wheel_center_x_' + csv_path)
    plt.clf()


def generate_diffVal_wheel_fig(wheel_parms, fig_path, csv_path):
    """
    生成最左车轮和第二个车轮的差值
    :param data: 散点图的数据
    :param fig_path: 生成散点图的路径
    :return:
    """
    # 创建一个空的dataframe
    wheel_data = pd.DataFrame(columns=["frame_index", "diffVal"])

    # 遍历车轮坐标列表，将数据放入 dataframe 中
    for frame in range(len(wheel_parms)):
        wheel_center_list = wheel_parms[frame]['wheel_center_list']
        wheel_center_list.sort()
        if (len(wheel_center_list) > 1):
            diffVal = wheel_center_list[1] - wheel_center_list[0]
        else:
            diffVal = 0
        wheel_data.loc[wheel_data.shape[0]] = {"frame_index": frame, "diffVal": diffVal}

    # 应用默认的默认seaborn主题，缩放比例和调色板
    sns.set()
    # Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
    data = wheel_data
    fig = sns.scatterplot(x=data['frame_index'], y=data['diffVal'],
                          data=data, s=50)
    scatter_fig = fig.get_figure()
    # fig_path为想要存入的文件夹或地址
    scatter_fig.savefig('diffVal_' + fig_path, dpi=400)
    wheel_data.to_csv('diffVal_' + csv_path)
    plt.clf()


generate_all_wheel_fig(wheel_parms, 'fig3.png', 'fig3.csv')
generate_left_wheel_fig(wheel_parms, 'fig3.png', 'fig3.csv')
generate_diffVal_wheel_fig(wheel_parms, 'fig3.png', 'fig3.csv')
