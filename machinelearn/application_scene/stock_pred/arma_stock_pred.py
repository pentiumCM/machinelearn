#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : arma_stock_pred.py
@Time    : 2020/8/4 22:07
@desc	 : ARMA - 实现股票预测
'''

from pandas import datetime

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.api import *
import tushare as ts

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

pro = ts.pro_api('6ecc6daaf139b983f0f83b196e21c34df880f9336ed6be286a19f5ea')  # 你的token
# data = ts.get_hist_data('600667')

# 数据采集时段
start = datetime.datetime(2000, 1, 1)
end = datetime.today()

data = pro.daily(ts_code='000001.SZ',start_date='20180701', end_date='20200805')

data.index = pd.to_datetime(data.index)
data = data.sort_index()

print(data.tail())

# 预测模型
arma = tsa.ARMA(data[['high']], (1, 1))
# 训练
model = arma.fit()
# 预测
predict = model.predict(1, 556)

figure = plt.figure('股票曲线', figsize=(8, 4))
ax = figure.add_axes([0.1, 0.1, 0.8, 0.8], xlabel='日期', ylabel='价格')
ax.plot(data[['high']], color=(1, 0, 0, 1), label='最高价格')

ax.plot(predict, color=(0, 0, 1, 1), label='预测最高价格')

plt.legend()
plt.show()
