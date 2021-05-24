#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : test.py
@Time    : 2021/5/20 10:18
@desc	 : 
'''

import torch
import torch.nn.functional as f


# 定义一个calss类继承pytouch
class Net(torch.nn.Module):
    # 初始化函数
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层神经元
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层神经元

    # 前向传播函数
    def forward(self, x):
        x = f.relu(self.hidden(x))  # 隐藏层函数计算后激励
        x = self.predict(x)  # 输出层计算后不激励
        return x


# 第一种搭建网络结构的方式
net1 = Net(1, 10, 1)

# 简单法搭建网络结构
net2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

# 查看两种建立网络的方法的区别
print(net1)

print(net2)
