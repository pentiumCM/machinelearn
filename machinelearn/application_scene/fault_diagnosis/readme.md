## 一、算法逻辑
+ 先执行 cnn_net_fault_diagnosis.py
+ 再执行 cnn_test.py
1. cnn_net_fault_diagnosis.py：<br>
该模块功能为：构建CNN(1D+2D)的网络结构，并且将网络结构保存为：cnn_model.h5模型文件，
提供给其他模块直接调用，进行模型训练。

2. cnn_test.py：<br>
该模块功能为：加载训练集数据，调用CNN的网络模构，进行训练

## 二、语法点：
+ 递减学习率：<br>
    + 递减学习率意义：<br>
    模型训练，关于超参数学习率的设置，一般初始设置为较大的值0.1,然后在学习的过程中进行缩小。
这样的好处：大的学习率帮助跳出局部最优值，小的学习率帮助模型收敛，有助于模型细化。
    
    + 实现方式：<br>
    采用 keras 的阶层性下降对学习率进行设置。代码中已设置。
    参考资料：https://blog.csdn.net/weixin_44791964/article/details/105334098?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522159452462419726869032536%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=159452462419726869032536&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v2~rank_blog_v1-1-105334098.pc_v2_rank_blog_v1&utm_term=%E5%AD%A6%E4%B9%A0%E7%8E%87
    
    + 实现效果：<br>
    Epoch 1/30
7000/7000 [==============================] - 39s 6ms/step - loss: 14.4728 - acc: 0.1003 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 2/30<br>
7000/7000 [==============================] - 30s 4ms/step - loss: 14.5063 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 3/30<br>
7000/7000 [==============================] - 23s 3ms/step - loss: 14.5063 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 00003: ReduceLROnPlateau reducing learning rate to 0.05000000074505806.<br>
Epoch 4/30<br>
7000/7000 [==============================] - 46s 7ms/step - loss: 14.5063 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 5/30<br>
7000/7000 [==============================] - 42s 6ms/step - loss: 14.5063 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 00005: ReduceLROnPlateau reducing learning rate to 0.02500000037252903.<br>
Epoch 6/30<br>
7000/7000 [==============================] - 39s 6ms/step - loss: 14.5063 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 7/30<br>
7000/7000 [==============================] - 31s 4ms/step - loss: 14.5063 - acc: 0.1000 - val_loss: 14.5063 - val_acc: 0.1000<br>
Epoch 00007: ReduceLROnPlateau reducing learning rate to 0.012500000186264515.<br>
    
    
+ concatenate：<br>
concatenate操作是网络结构设计中很重要的一种操作，经常用于将特征联合，多个卷积特征提取框架提取的特征融合或者是将输出层的信息进行融合，而add层更像是信息之间的叠加。

+ Flatten层：<br>
用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡