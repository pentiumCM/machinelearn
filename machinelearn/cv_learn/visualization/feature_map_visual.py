#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : feature_map_visual.py
@Time    : 2020/12/7 11:12
@desc	 : feature map可视化
'''

import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle

from sys import path

path.append('/residual model path')
import residual_model
from residual_model import Residual_Model

model = Residual_Model()
model.load_state_dict(torch.load('./model.pkl'))


class myNet(nn.Module):
    def __init__(self, pretrained_model, layers):
        super(myNet, self).__init__()
        self.net1 = nn.Sequential(*list(pretrained_model.children())[:layers[0]])
        self.net2 = nn.Sequential(*list(pretrained_model.children())[:layers[1]])
        self.net3 = nn.Sequential(*list(pretrained_model.children())[:layers[2]])

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net(out1)
        out3 = self.net(out2)
        return out1, out2, out3


def get_features(pretrained_model, x, layers=[3, 4, 9]):  ## get_features 其实很简单
    '''
    1.首先import model 
    2.将weights load 进model
    3.熟悉model的每一层的位置，提前知道要输出feature map的网络层是处于网络的那一层
    4.直接将test_x输入网络，*list(model.chidren())是用来提取网络的每一层的结构的。net1 = nn.Sequential(*list(pretrained_model.children())[:layers[0]]) ,就是第三层前的所有层。
    
    '''
    net1 = nn.Sequential(*list(pretrained_model.children())[:layers[0]])
    #   print net1  
    out1 = net1(x)

    net2 = nn.Sequential(*list(pretrained_model.children())[layers[0]:layers[1]])
    #   print net2  
    out2 = net2(out1)

    # net3 = nn.Sequential(*list(pretrained_model.children())[layers[1]:layers[2]])  
    # out3 = net3(out2)  

    return out1, out2


with open('test.pickle', 'rb') as f:
    data = pickle.load(f)
x = data['test_mains'][0]
x = Variable(torch.from_numpy(x)).view(1, 1, 128, 1)  ## test_x必须为Varibable
# x = Variable(torch.randn(1,1,128,1))
if torch.cuda.is_available():
    x = x.cuda()  # 如果模型的训练是用cuda加速的话，输入的变量也必须是cuda加速的，两个必须是对应的，网络的参数weight都是用cuda加速的，不然会报错
    model = model.cuda()
output1, output2 = get_features(model, x)  ## model是训练好的model,前面已经import 进来了Residual model
print('output1.shape:', output1.shape)
print('output2.shape:', output2.shape)
# print('output3.shape:',output3.shape)
output_1 = torch.squeeze(output2, dim=0)
output_1_arr = output_1.data.cpu().numpy()  # 得到的cuda加速的输出不能直接转变成numpy格式的，当时根据报错的信息首先将变量转换为cpu的，然后转换为numpy的格式
output_1_arr = output_1_arr.reshape([output_1_arr.shape[0], output_1_arr.shape[1]])
