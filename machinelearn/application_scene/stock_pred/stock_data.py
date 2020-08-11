#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : stock_data.py
@Time    : 2020/8/4 15:06
@desc	 : 股票数据
'''

import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime

import pandas_datareader.data as web


def data_test():
    # 一、参数设置
    # 股票数据源
    data_source = 'yahoo'
    # 股票名称，规则见同级 md 说明文件
    stock_name = '512480.SS'
    # 数据采集时段
    start = datetime.datetime(2000, 1, 1)
    end = datetime.today()

    # 从雅虎财经读入2019年至今的半导体数据
    stock_df = web.DataReader(stock_name, data_source, start, end)
    pd.set_option('display.max_columns', None)  # 显示所有列

    print(stock_df.head())
    print(stock_df.tail())

    # 绘制stock历史收盘价trend图
    plt.figure(figsize=(16, 8))
    plt.plot(stock_df['Close'])
    plt.show()


def prepare_stock_data(stock_name, data_source, start_time, end_time):
    """
    采集股票数据

    :param stock_name: 股票代码
    :param data_source:  股票数据源，如 yahoo
    :param start_time: 股票数据开始时间
    :param end_time:  股票数据结束时间
    :return:    股票的表格数据
    """
    stock_df = web.DataReader(stock_name, data_source=data_source, start=start_time, end=end_time)
    return stock_df
