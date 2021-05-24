#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : json_txt2excel.py
@Time    : 2021/1/12 20:41
@desc	 : txt 中的 json 转 excel
'''
import json
import os

import pandas as pd

file_dir = 'F:/数据采集/list/'

file_dist = 'F:/数据采集/数据表格/'

for root, dirs, files in os.walk(file_dir):

    for item in files:
        file_path = file_dir + item
        fopen = open(file_path, 'r')
        lines = fopen.readline()

        result = json.loads(lines)
        records = result['rows']

        record_df = pd.DataFrame(records)
        record_df.to_excel(file_dist + item + ".xlsx", index=False)  # openpyxl要安装这个模块，后就不报错了，
