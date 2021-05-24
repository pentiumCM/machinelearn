#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : productor.py
@Time    : 2021/4/29 20:34
@desc	 : 消息队列——生产者
'''

import pika
import json

host_ip = '192.168.142.63'
port = 5672

virtual_host = '/test'
QUEUE_NAME = "hello"

# mq用户名和密码
credentials = pika.PlainCredentials('admin', 'admin')

# 1. 建立连接
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=host_ip, port=port, virtual_host=virtual_host, credentials=credentials))

# 2. 建立通道
channel = connection.channel()

# 3. 指定队列
channel.queue_declare(queue=QUEUE_NAME, durable=True)

# 4. 发布消息
message = json.dumps({'name': 'pentiumcm'})
# 向队列插入数值 routing_key是队列名
channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=message)

print("生产者发送完成：" + message)

connection.close()
