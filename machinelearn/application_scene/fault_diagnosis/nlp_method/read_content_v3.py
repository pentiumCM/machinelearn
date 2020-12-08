#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : read_content.py
@Time    : 2020/11/5 21:15
@desc	 : 读取文本内容，将word中的数据转换，进行拆分
'''

import re
import pandas as pd


def txt2dict(txt_name, extract_node, csv_savename):
    """
    文本数据提取成结构化数据，并存储在csv中
    :param txt_name: 读取的文本文件
    :param extract_node: 按节点提取对应文本中的内容
    :param csv_savename: 存储成 csv 文件数据
    :return:
    """

    with open(txt_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 文本过滤格式：去除开头制表符，中文冒号改英文
    for i in range(len(lines)):
        filter_str = ''.join(lines[i].split('\t'))
        filter_str = filter_str.replace('：', ':')

        lines[i] = filter_str

    result = []
    rows = len(lines)

    for begin_index, line in enumerate(lines):
        if line[0].isdigit():  # 以数字开头代表：故障结论

            # 利用字典存放每个单元的文本数据
            result_item = {}

            # 寻找下一个数字开头的位置，则【begin_index，end_index】为当前一段故障数据的单元区间
            end_index = begin_index
            while end_index < rows - 1 and (not lines[end_index + 1][0].isdigit()):
                end_index += 1
            # 将 end_index 定位到下一单元的开头，方便进行提取
            if (end_index < rows):
                end_index += 1

            # 提取故障结论
            try:
                conclusion = line.split('、')[1]
                print(begin_index, conclusion)
            except:
                print(begin_index, begin_index)

            result_item['conclusion'] = conclusion

            # 提取每个节点起始的位置索引
            index_list = []
            for node in extract_node:
                for index in range(begin_index, end_index):
                    if lines[index].startswith(node):
                        index_list.append(index)
            index_list.sort()

            # 结构化提取的规则：根据提取的节点的位置，然后分别提取每个部分文本内容
            for index_num in range(len(index_list)):
                # 【index1 : index2】之间的数据为某一节点的数据
                index1 = 0
                index2 = 0
                if (index_num < len(index_list) - 1):
                    index1 = index_list[index_num]
                    index2 = index_list[index_num + 1]
                else:
                    index1 = index_list[len(index_list) - 1]
                    index2 = end_index

                # 提取当前节点内的数据，为【index1 : index2】
                # 1. 处理值和节点在同一行的情况，如，症状：每个球的位置上有椭圆形的磨痕
                content = ''
                if lines[index1].split(':')[1]:
                    content += lines[index1].split(':')[1]
                # 2. 直接循环附加
                for num in range(index1 + 1, index2):
                    content += lines[num]

                result_item[extract_node[index_num]] = content

            result.append(result_item)

    df = pd.DataFrame(result)

    df.to_csv(csv_savename, index=None, encoding='utf_8_sig')


if __name__ == '__main__':
    txt_filename = 'newdataFromHtmlText.txt'
    csv_savename = 'fault_analysis_v2.csv'

    extract_node = ['症状', '原因', '补救方法']

    txt2dict(txt_filename, extract_node, csv_savename)
