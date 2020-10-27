#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : lstm_stock_pred.py
@Time    : 2020/8/3 20:18
@desc	 : 使用 LSTM 预测股票价格

'''


# 训练过程：
    # 取一定时间点的数据（如50个交易日的数据）作为输入，
    # 预测该段时间的下一个交易日的收盘价，不断缩小真实收盘价与预测收盘价的差值loss即可。
# (1) 加载stock数据
# (2) 构造训练数据
# (3) LSTM建模
# (4) 预测stock
# (5) 查看stock trend拟合效果


import pandas as pd

# from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt

from pandas import datetime
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

import quandl

# 加载数据
start = datetime.datetime(2000, 10, 12)
# start = datetime.datetime(1971, 1, 1)
end = datetime.today()

quandl.ApiConfig.api_key = "QLczKWkHuD_zpxstvf4m"  # YOURAPIKEY换成你的API key
google_stock = pd.DataFrame(quandl.get("WIKI/GOOGL", start_date=start, end_date=end))
print(google_stock.shape)

print(google_stock.tail())
print(google_stock.head())

# 绘制stock历史收盘价trend图
plt.figure(figsize=(16, 8))
plt.plot(google_stock['Close'])
plt.show()

# 构造训练集与验证集
# 时间点长度
time_stamp = 50

# 划分训练集与验证集
google_stock = google_stock[['Open', 'High', 'Low', 'Close', 'Volume']]  # 'Volume'
train = google_stock[0:2800 + time_stamp]
valid = google_stock[2800 - time_stamp:]

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)
x_train, y_train = [], []

# 训练集
print(scaled_data.shape)
print(scaled_data[1, 3])
for i in range(time_stamp, len(train)):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i, 3])

x_train, y_train = np.array(x_train), np.array(y_train)

# 验证集
scaled_data = scaler.fit_transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)):
    x_valid.append(scaled_data[i - time_stamp:i])
    y_valid.append(scaled_data[i, 3])

x_valid, y_valid = np.array(x_valid), np.array(y_valid)

print(x_train.shape)
print(x_valid.shape)
train.head()

# 创建并训练LSTM模型
# 超参数
epochs = 3
batch_size = 16
# LSTM 参数: return_sequences=True LSTM输出为一个序列。默认为False，输出一个值。
# input_dim：输入单个样本特征值的维度
# input_length：输入的时间点长度
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

# 预测stock价格
closing_price = model.predict(x_valid)
scaler.fit_transform(pd.DataFrame(valid['Close'].values))
# 反归一化
closing_price = scaler.inverse_transform(closing_price)
y_valid = scaler.inverse_transform([y_valid])
# print(y_valid)
# print(closing_price)
rms = np.sqrt(np.mean(np.power((y_valid - closing_price), 2)))
print(rms)
print(closing_price.shape)
print(y_valid.shape)

# 拟合stock trend
plt.figure(figsize=(16, 8))
dict_data = {
    'Predictions': closing_price.reshape(1, -1)[0],
    'Close': y_valid[0]
}
data_pd = pd.DataFrame(dict_data)

plt.plot(data_pd[['Close', 'Predictions']])
plt.show()

print("hello")