#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : stock_cost.py
@Time    : 2020/9/18 13:54
@desc	 : 
'''

import pandas as pd

df = pd.read_excel('D:/money/stock_deal.xlsx',sheet_name='证券')

buy_price_in = df['买入价格'].values

buy_price_out = df['卖出价格'].values

print(df.head())
