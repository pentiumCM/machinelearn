#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : __init__.py.py
@Time    : 2020/9/5 8:51
@desc	 : 
'''

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

# 应用默认的默认seaborn主题，缩放比例和调色板
sns.set()

# 加载数据集
# tips = sns.load_dataset("tips.csv")
# Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
tips = pd.read_csv("F:/project/tx/filter/wheel_6.csv")

# 绘制散点图
fig = sns.scatterplot(x=tips['frame_index'], y=tips['wheel_centers'],
                      data=tips, s=50)
plt.show()

