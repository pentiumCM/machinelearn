#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : add.py
@Time    : 2021/3/23 10:26
@desc	 : 
'''
import json

if __name__ == '__main__':
    # jsonData = '{"imgpath":"F:/project/jit/MrZ/examtestmis/code/yolov5-3.0/yolov5-exam/inference/images2"}'
    jsonData = '{"imgpath":"F:/project/jit/MrZ/examtestmis/code/yolov5-3.0/yolov5-exam/inference/images2"}'
    text = json.loads(jsonData)
    res = text['imgpath']
    print(res)
