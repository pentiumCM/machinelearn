# 搭建神经网络基本流程
##一、基础知识：
###1.什么是人工神经网络:
    
    人工神经元又称为感知机，输入经过加权和偏置后，由激活函数处理后决定输出。
    
    由大量的人工神经元互相连接而形成的复杂网络结构称为人工神经网络（Artificial Neural Network, ANN）

![avatar](https://img-blog.csdn.net/20180607170423382?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

###2.神经网络的相关概念:
####2.1 输入层、隐含层，输出层

    一般的神经网络包含以下几个部分，输入层，隐含层和输出层。
    隐层的层数越多，隐层结点数目越多，在非线性的激活函数下，神经网络就可以学习更深层次的特征。

![avatar](https://img-blog.csdn.net/20180607182202932?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

####2.2 激活函数

    激活函数是神经网络设计的核心单元，激活函数是用来加入非线性因素的，因为线性模型的表达能力不够。
    激活函数需要满足一下几个条件：
    1.非线性。如果激活函数是线性的，那么不管引入多少隐层，其效果和单层感知机没有任何差别。
    2.可微性。训练网路时使用的基于梯度的优化方法需要激活函数必须可微。
    3.单调性。单调性保证了神经网络模型简单。


 
####2.3 权重和偏置
![avatar](https://img-blog.csdn.net/20180607174206895?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    如上图所示，假设神经元输入是X = [x1,x2,x3]，权重为w,偏置为b, 那么其输出为：
![avatar](https://img-blog.csdn.net/20180607174352811?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    可以看出,w，b会影响感知机的输出结果，正是因为此神经网络在迭代过程中修改的是权重w和偏置b。
####2.4 损失函数
    损失函数（loss function）也叫代价函数（cost function）。是神经网络优化的目标函数，神经网络训练或者优化的过程就是最小化损失函数的过程（损失函数值小了，对应预测的结果和真实结果的值就越接近）。
    
    损失函数也有很多种，拿常用的交叉熵损失函数举例，其单个样本二分类的损失函数公式如下：

    当期望输出为1时，，此时希望损失函数尽量的小，那么就要接近1；

    当期望输出为0时，，此时希望损失函数尽量的小，那么就要接近0。

    对于m样本的损失函数可以使用：
![avater](https://img-blog.csdn.net/20180607184513456?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

####2.5 反向传播（Back Propagation,BP）算法
    BP算法分为正向传播和误差的反向传播两个部分。
    正向正向传播时，输入样本从输入层进入网络，经隐层逐层传递至输出层，如果输出层的实际输出与期望输出(导师信号)不同，则转至误差反向传播；如果输出层的实际输出与期望输出(导师信号)相同，结束学习算法。
    反向反向传播时，将输出误差(期望输出与实际输出之差)按原通路反传计算，通过隐层反向，直至输入层，在反传过程中将误差分摊给各层的各个单元，获得各层各单元的误差信号，并将其作为修正各单元权值的根据。这一计算过程使用梯度下降法完成，在不停地调整各层神经元的权值和阈值后，使误差信号减小到最低限度。
    权值和阈值不断调整的过程，就是网络的学习与训练过程，经过信号正向传播与误差反向传播，权值和阈值的调整反复进行，一直进行到预先设定的学习训练次数，或输出误差减小到允许的程度。
    如图所示，其中蓝色为正向传播，红色为反向传播，反向传播的导数为损失函数对各变量求导。
![avater](https://img-blog.csdn.net/20180607194629988?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)