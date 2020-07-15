#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : memory_size.py
@Time    : 2020/5/14 17:08
@desc	 : python 各个数据类型所占的内存
'''

import numpy as np
import sys

# 32位整型
ai32 = np.array([], dtype=np.int32)
bi32 = np.arange(1, dtype=np.int32)
ci32 = np.arange(5, dtype=np.int32)

# 64位整型
ai64 = np.array([], dtype=np.int64)
bi64 = np.arange(1, dtype=np.int64)
ci64 = np.arange(5, dtype=np.int64)

# 16位浮点数
af16 = np.asarray([],dtype=np.float16)
bf16 = np.arange(1, dtype=np.float16)

# 32位浮点数
af32 = np.array([], dtype=np.float32)
bf32 = np.arange(1, dtype=np.float32)
cf32 = np.arange(5, dtype=np.float32)

# 64位浮点数
af64 = np.array([], dtype=np.float64)
bf64 = np.arange(1, dtype=np.float64)
cf64 = np.arange(5, dtype=np.float64)

print("size of 0 int32 number: %f" % sys.getsizeof(ai32))
print("size of 1 int32 number: %f" % sys.getsizeof(bi32))
print("size of 5 int32 numbers: %f" % sys.getsizeof(ci32), end='\n\n')

print("size of 0 int64 number: %f" % sys.getsizeof(ai64))
print("size of 1 int64 number: %f" % sys.getsizeof(bi64))
print("size of 5 int64 numbers: %f" % sys.getsizeof(ci64), end='\n\n')

print("size of 0 float16 number: %f" % sys.getsizeof(af16))
print("size of 1 float16 number: %f" % sys.getsizeof(bf16),end='\n\n')

print("size of 0 float32 number: %f" % sys.getsizeof(af32))
print("size of 1 float32 number: %f" % sys.getsizeof(bf32))
print("size of 5 float32 numbers: %f" % sys.getsizeof(cf32), end='\n\n')

print("size of 0 float64 number: %f" % sys.getsizeof(af64))
print("size of 1 float64 number: %f" % sys.getsizeof(bf64))
print("size of 5 float64 numbers: %f" % sys.getsizeof(cf64))
