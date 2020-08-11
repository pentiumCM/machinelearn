#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : stock_prediction.py
@Time    : 2020/8/3 16:34
@desc	 : 股票预测 - RNN实现
'''

import numpy as np
import pandas as  pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

import keras
from utils.scaler_utils import normalized_scaler_transdata
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

from sklearn import preprocessing

from pandas import datetime
import pandas_datareader.data as web

# 引入参数
stock_name = '^GSPC'

# 滑窗的大小为22，预测时，滑窗拿前21天的数据（包括Adjusted Close)预测第22天的数据
seq_len = 22
d = 0.2
shape = [4, seq_len, 1]  # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 300


# 下载数据，并且正则化
def get_stock_data(stock_name, normalize=True):
    start = datetime.datetime(1971, 1, 1)
    # end = datetime.date.today()
    end = datetime.today()
    df = web.DataReader(stock_name, "yahoo", start, end)
    df.drop(['Volume', 'Close'], 1, inplace=True)

    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
    return df


df = get_stock_data(stock_name, normalize=True)


# Adj Close是我们的标签列，我们预测的是最后一列的数据


# 绘制正则化后的标签数据
def plot_stock(stock_name):
    df = get_stock_data(stock_name, normalize=True)
    print(df.tail())
    plt.plot(df['Adj Close'], color='red', label='Adj Close')
    plt.legend(loc='best')
    plt.show()


plot_stock(stock_name)


# 把最后一天的Adjusted Close作为y值
def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.values
    sequence_length = seq_len + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):  # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length])  # index : index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0])  # 90% split

    train = result[:int(row), :]  # 90% date
    X_train = train[:, :-1]  # all data until day m
    y_train = train[:, -1][:, -1]  # day m + 1 adjusted close price

    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))

    return [X_train, y_train, X_test, y_test]


X_train, y_train, X_test, y_test = load_data(df, seq_len)


# 搭建神经网络
def build_model2(layers, neurons, d):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[2]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(neurons[2], kernel_initializer="uniform", activation='relu'))
    model.add(Dense(neurons[3], kernel_initializer="uniform", activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model = build_model2(shape, neurons, d)

# 运行模型
history = model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=epochs,
    validation_split=0.1,
    verbose=1)


# 可视化 loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

print("hello")
