#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : feature_map_visual.py
@Time    : 2020/12/7 11:12
@desc	 : feature map可视化
            参考：https://blog.csdn.net/weixin_36411839/article/details/109097714
'''

import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training and testing datasets.
pic_dir = 'set06_V003_I00983_visible.jpg'

# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()


# 单张图像送入
# 构建网络
# 提取中间层
# 可视化特征图

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)


def get_picture_rgb(picture_dir):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    skimage.io.imsave('new4.jpg', img256)

    # 取单一通道值显示
    # for i in range(3):
    #     img = img256[:,:,i]
    #     ax = plt.subplot(1, 3, i + 1)
    #     ax.set_title('Feature {}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(img)

    # r = img256.copy()
    # r[:,:,0:2]=0
    # ax = plt.subplot(1, 4, 1)
    # ax.set_title('B Channel')
    # # ax.axis('off')
    # plt.imshow(r)

    # g = img256.copy()
    # g[:,:,0]=0
    # g[:,:,2]=0
    # ax = plt.subplot(1, 4, 2)
    # ax.set_title('G Channel')
    # # ax.axis('off')
    # plt.imshow(g)

    # b = img256.copy()
    # b[:,:,1:3]=0
    # ax = plt.subplot(1, 4, 3)
    # ax.set_title('R Channel')
    # # ax.axis('off')
    # plt.imshow(b)

    # img = img256.copy()
    # ax = plt.subplot(1, 4, 4)
    # ax.set_title('image')
    # # ax.axis('off')
    # plt.imshow(img)

    img = img256.copy()
    ax = plt.subplot()
    ax.set_title('new-image')
    # ax.axis('off')
    plt.imshow(img)

    plt.show()


class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''

    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # input_size=(3*256*256)，padding=2
            nn.ReLU(),  # input_size=(32*256*256)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(32*128*128)
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),  # input_size=(32*128*128)
            nn.ReLU(),  # input_size=(64*128*128)
            nn.MaxPool2d(2, 2)  # output_size=(64*64*64)
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),  # 进行线性变换
            nn.ReLU()  # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 62)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        print('---------', self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            print(module)
            x = module(x)
            print('name', name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


def get_feature():  # 特征可视化
    # 输入数据
    img = get_picture(pic_dir, transform)  # 输入的图像是【3,256,256】
    # 插入维度
    img = img.unsqueeze(0)  # 【1,3,256,256】
    img = img.to(device)

    # 特征输出
    net = LeNet().to(device)
    # net.load_state_dict(torch.load('./model/net_050.pth'))
    exact_list = ['conv1']
    myexactor = FeatureExtractor(net, exact_list)  # 输出是一个网络
    x = myexactor(img)

    # 特征输出可视化
    for i in range(32):  # 可视化了32通道
        ax = plt.subplot(6, 6, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        ax.set_title('new—conv1-image')

        plt.imshow(x[0].data.cpu().numpy()[0, i, :, :], cmap='jet')

    plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型


# 训练
if __name__ == "__main__":
    # get_picture_rgb(pic_dir)
    get_feature()
