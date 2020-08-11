#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : uuid_utils.py
@Time    : 2020/8/6 11:44
@desc	 : uuid 工具模块
'''

# coding=utf-8

import uuid

name = 'test_name'
# namespace = 'test_namespace'
namespace = uuid.NAMESPACE_URL

print(uuid.uuid1())
print(uuid.uuid3(namespace, name))
print(uuid.uuid4())
print(uuid.uuid5(namespace, name))
