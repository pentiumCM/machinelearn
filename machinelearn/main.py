#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : feature_extraction_VGG16.py
@Time    : 2019/10/21 14:41
@desc	 : 程序的入口
'''

import math


# for num in range(200, 300, step=15):  # 迭代 10 到 20 之间的数字


def generate_sub_analysis_interval(start, end, step):
    """
    产生子区间的分析区间
    :param start: 区间开始位置
    :param end: 区间结束位置
    :param step: 步长
    :return:
    """
    sub_analysis_interval_list = []
    interval_num = math.ceil((end - start) / step)

    for item in range(interval_num):
        # 判断最后一个小区间是否越过区间结束位置，越过直接用结束位置赋值
        if (start + step * (item + 1) <= end):
            sub_analysis_interval_list_item = [start + step * item, start + step * (item + 1)]
        else:
            sub_analysis_interval_list_item = [start + step * item, end]
        sub_analysis_interval_list.append(sub_analysis_interval_list_item)

    return sub_analysis_interval_list


if __name__ == '__main__':
    print(generate_sub_analysis_interval(200, 300, 15))
