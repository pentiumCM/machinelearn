#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : save_fig.py
@Time    : 2020/9/6 15:10
@desc	 : 
'''

import seaborn as sns
import pandas as pd

# 应用默认的默认seaborn主题，缩放比例和调色板
sns.set()

# Seaborn 要求原始数据的输入类型为 pandas 的 Dataframe 或 Numpy 数组
data = pd.read_csv("E:/tx/tips.csv")

fig_name = 'scatterplot1.png'

# fig_path为想要存入的文件夹或地址
fig = sns.scatterplot(x=data['total_bill'], y=data['tip'], hue='sex',
                      data=data, palette='Set1', s=100)
scatter_fig = fig.get_figure()
scatter_fig.savefig(fig_name, dpi=400)
