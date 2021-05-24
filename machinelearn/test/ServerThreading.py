#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : ServerThreading.py
@Time    : 2021/3/23 10:03
@desc	 : 
'''
import json
import socket
import threading


def main():
    # 1. 初始化服务端socket
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名称
    # host = socket.gethostname()
    host = '127.0.0.1'
    # 设置一个端口
    port = 9001
    # 2. bind：将套接字与本地主机和端口绑定
    serversocket.bind((host, port))

    # 3. listen：设置监听最大连接数
    serversocket.listen(5)

    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
    print("服务器地址:%s" % str(myaddr))

    # 循环等待接受客户端信息
    while True:
        # 4. accept：获取一个客户端连接
        clientsocket, addr = serversocket.accept()
        print("连接地址:%s" % str(addr))
        try:
            # 5. recive：为每一个请求开启一个处理线程
            t = ServerThreading(clientsocket)
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass

    # 7. close
    serversocket.close()
    pass


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程.....")
        try:
            # 接受数据
            msg = ''
            while True:
                # 读取 recvsize 个字节
                rec = self._socket.recv(self._recvsize)
                datalen = len(rec)
                if datalen == 0:
                    break
                # 解码
                msg += rec.decode(self._encoding)

                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                # if msg.strip().endswith('over'):
                #     msg = msg[:-4]
                #     break

            msg = json.loads(msg)
            image_path = msg["imgpath"]

            # 发送数据
            dict = {"name": "Tom", "age": image_path}  # 字典
            res = json.dumps(dict)
            self._socket.send(("%s" % res).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")

        pass

    def __del__(self):
        pass


if __name__ == "__main__":
    main()
