#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : generateTXT.py
@Time    : 2020/9/16 17:51
@desc	 : 生成4个txt文件：指定训练集、验证集、数据集、训练验证集
'''

import os
import random
import time

xmlfilepath = 'F:/experiment/Caltech/VOC_process/Annotations'
saveBasePath = 'F:/experiment/Caltech/VOC_process/ImageSets/Main'

if not os.path.exists(saveBasePath):
    os.makedirs(saveBasePath)
# 设置训练集和测试集的百分比
trainval_percent = 0.5
train_percent = 0.5
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)  # xml文件的数量
index_list = range(num)  # 生成一个index列表
trainval_num = int(num * trainval_percent)
train_num = int(trainval_num * train_percent)
trainval_index = random.sample(index_list, trainval_num)
train_index = random.sample(trainval_index, train_num)

print("train and val size", trainval_num)
print("train size", train_num)

ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

# Start time
start = time.time()
for i in index_list:
    name = os.path.splitext(total_xml[i])[0] + '\n'
    if i in trainval_index:
        ftrainval.write(name)
        if i in train_index:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
# End time
end = time.time()
seconds = end - start
print("Time taken : {0} seconds".format(seconds))
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
