#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : test.py
@Time    : 2020/11/1 17:00
@desc	 : pytorch 测试文件
'''

import torch

a = torch.rand(5, 3)

print(a)

zeros = torch.zeros(5, 3, dtype=torch.long)
print(zeros)

tensors = torch.tensor([5.5, 3])
print(tensors)

x = tensors.new_ones(5, 3, dtype=torch.double)

# new_* methods take in sizes
print(x)

print("size：", x.size())
