#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : consumer.py
@Time    : 2021/4/29 20:34
@desc	 : 消息队列——消费者
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
connection = pika.BlockingConnection(pika.ConnectionParameters(host=host_ip, port=port,
                                                               virtual_host=virtual_host,
                                                               credentials=credentials))
# 2. 建立通道
channel = connection.channel()

# 3. 指定队列
channel.queue_declare(queue=QUEUE_NAME, durable=True)


# 定义一个回调函数来处理消息队列中的消息
def callback(ch, method, properties, body):
    # 接收到生产者消息
    msg = json.loads(body)
    print("消费者接收:" % msg['name'])


# 4. 消费消息
channel.basic_consume(queue=QUEUE_NAME, auto_ack=True, on_message_callback=callback)

channel.start_consuming()
