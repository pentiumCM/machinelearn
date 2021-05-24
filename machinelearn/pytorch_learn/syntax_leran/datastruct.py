#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : datastruct.py
@Time    : 2021/5/24 11:30
@desc	 : pytorch 数据结构
'''

import torch

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

print(help(torch.add))

a.add(1)
print("a：", a)
print("b：", b)

x = torch.randn(1)
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)  # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # `.to`也能在移动时改变dtype



print(dir(torch.autograd))
