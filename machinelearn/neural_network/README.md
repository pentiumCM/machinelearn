# 神经网络体系搭建
## 一、基础知识：
### 1.什么是人工神经网络:
    人工神经元又称为感知机，输入经过加权和偏置(阈值)后，由激活函数处理后决定输出。
    (偏置值允许将激活函数向左或向右移位，这可能是成功学习的关键。）

    由大量的人工神经元互相连接而形成的复杂网络结构称为人工神经网络（Artificial Neural Network, ANN）

![avatar](https://img-blog.csdn.net/20180607170423382?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 2.神经网络的相关概念:

![pic](https://upload-images.jianshu.io/upload_images/4964755-925cfc2ba47c214b.png?imageMogr2/auto-orient/strip|imageView2/2)
#### 2.1 输入层、隐含层，输出层

    一般的神经网络包含以下几个部分，输入层，隐含层和输出层。
    
    隐层的层数越多，隐层结点数目越多，在非线性的激活函数下，神经网络就可以学习更深层次的特征。

![avatar](https://img-blog.csdn.net/20180607182202932?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 2.2 激活函数

    激活函数是神经网络设计的核心单元，激活函数是用来加入非线性因素的，因为线性模型的表达能力不够。
    
    激活函数需要满足一下几个条件：
    1.非线性。如果激活函数是线性的，那么不管引入多少隐层，其效果和单层感知机没有任何差别。
    2.可微性。训练网路时使用的基于梯度的优化方法需要激活函数必须可微。
    3.单调性。单调性保证了神经网络模型简单。
    
#### 2.3 权重和偏置
![avatar](https://img-blog.csdn.net/20180607174206895?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    如上图所示，假设神经元输入是X = [x1,x2,x3]，权重为w,偏置为b, 那么其输出为：
![avatar](https://img-blog.csdn.net/20180607174352811?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

    可以看出,w，b会影响感知机的输出结果，正是因为此神经网络在迭代过程中修改的是权重w和偏置b。
#### 2.4 损失函数
    损失函数（loss function）也叫代价函数（cost function）。是神经网络优化的目标函数，神经网络训练或者优化的过程就是最小化损失函数的过程（损失函数值小了，对应预测的结果和真实结果的值就越接近）。
    
    求损失函数最小值的方法通常为梯度下降。
    
    损失函数也有很多种，拿常用的交叉熵损失函数举例，其单个样本二分类的损失函数公式如下：

    当期望输出为1时，，此时希望损失函数尽量的小，那么就要接近1；

    当期望输出为0时，，此时希望损失函数尽量的小，那么就要接近0。

    对于m样本的损失函数可以使用：
![avater](https://img-blog.csdn.net/20180607184513456?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 2.5 反向传播（Back Propagation,BP）算法
    BP算法分为正向传播和误差的反向传播两个部分。
    
    正向正向传播时，输入样本从输入层进入网络，经隐层逐层传递至输出层，如果输出层的实际输出与期望输出(导师信号)不同，则转至误差反向传播；如果输出层的实际输出与期望输出(导师信号)相同，结束学习算法。
    
    反向传播时，将输出误差(期望输出与实际输出之差)按原通路反传计算，通过隐层反向，直至输入层，在反传过程中将误差分摊给各层的各个单元，获得各层各单元的误差信号，并将其作为修正各单元权值的根据。这一计算过程使用梯度下降法完成，在不停地调整各层神经元的权值和阈值后，使误差信号减小到最低限度。
    
    权值和阈值不断调整的过程，就是网络的学习与训练过程，经过信号正向传播与误差反向传播，权值和阈值的调整反复进行，一直进行到预先设定的学习训练次数，或输出误差减小到允许的程度。
    
    如图所示，其中蓝色为正向传播，红色为反向传播，反向传播的导数为损失函数对各变量求导。
![avater](https://img-blog.csdn.net/20180607194629988?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
---

## 二、数学公式及代码实现：
本文以实现逻辑回归(本质上为二分类问题)为例，逻辑回归如图所示，只有一个神经元结点。
![avater](https://img-blog.csdn.net/2018061011100253?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 1. 激活函数

logistic回归的激活函数一般使用sigmoid(x),其他情况可以使用tanh(x),ReLU(x)或者泄露ReLU(x)。
这里以sigmoid(x)为例表达式如下：

![avater](https://img-blog.csdn.net/20180609165424734?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```python
def sigmoid(z):
    """
    sigmoid激活函数
    :param z: 输入
    :return: sigmoid(z)
    """
    return 1/(1 + np.exp(-z))
```

### 2.初始化参数
神经网络中有两个参数需要初始化分别是权重w和偏置b，初始化的方式可以全部初始化为0，也可以随机初始化。<br>
需要注意到的是对于logistic回归，可以将权重初始化为零，但是对于神经网络的歌参数数组全部初始化为零，再使用梯度下降那么就会无效。原因是无论哪个初始输入与零权重相乘的到的结果也全是零，从而激活函数输出也是一样的，最终导致反向传播的结果也是一样。由于本文实现的是逻辑回归，故采用零值初始化。

```python
def initializeParamWithZeros(dim):
    """
    初始化权重和偏置
    :param dim: 输入维度
    :return: 返回初始化的w和b
              w:(dim,1)的向量
              b:标量
    """
    w = np.zeros((dim,1))
    b = 0
    return w,b
```

### 3.BP算法

BP算法分为两个部分正向传播与反向传播。<br>
逻辑回归中正向传播用公式表示为：

![avater](https://img-blog.csdn.net/20180610112105596?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
其中X是输入数据，A是输出数据。<br>

反向传播是采用梯度下降法使误差函数减小，误差函数表示为：

![pic](https://img-blog.csdn.net/20180610113135722?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
其中m为输入数据数目，![pic](https://img-blog.csdn.net/20180610113251246?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)是输入真实标签，![pic](https://img-blog.csdn.net/20180610113353575?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)是激活函数输出，即预测值。<br>

反向传播过程中使用梯度下降来实现损失函数的减小，需要先求得损失函数J对w和b的偏导分别为:

![pic](https://img-blog.csdn.net/20180610115507220?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
：全局成本函数对w的导数，也可以理解为各个样本损失函数对w导数的平均值

![pic](https://img-blog.csdn.net/20180610115515278?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
：全局成本函数对b的导数，也可以理解为各个样本损失函数对b导数的平均值

```python
def BackPropagate(w,b,X,Y):
    """
    BP算法
    :param w: 权重
    :param b: 偏置
    :param X: 输入数据
    :param Y: 输入标签
    :return: 梯度和损失函数
    """
    #输入数据数目
    num = Y.shape[0]
 
    #正向传播
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -1 / num * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
 
    #反向传播
    dw = 1 / num * np.dot(X,(A - Y).T)
    db = 1 / num * np.sum(A - Y)
 
    #用字典存储dw和db
    gradients = {"dw": dw,
             "db": db}
    return gradients,cost
```

### 4. 梯度下降与优化
神经网络的目标是针对参数W和b来求其损失函数J(W,b)的最小值.<br>
为了求解最优的W和b,我们可以重复梯度下降法的迭代步骤来求解最优的W和b，W和b更新公式为:

![pic](https://img-blog.csdn.net/20180610153302439?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

其中α为学习速率。

### 5.预测

训练好后的参数就可以在实际应用中使用了。<br>
预测结果分为两部分，第一部分计算输出：

![pic](https://img-blog.csdn.net/20180610162244774?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzM1ODIxOTc2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

然后判断输出和0.5的大小，大于0.5则为1，反之则为0

```python
def predict(w,b,X,Y):
    """
    :param w: 训练后的权值
    :param b: 训练后偏置
    :param X: 输入数据
    :param Y: 输入数据的标签
    :return: 预测概率大于0.5的
    """
    #获取输入数目m
    m = Y.shape[0]
    #存储预测结果
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    #神经元输出
    A = sigmoid(np.dot(w.T,X) + b)
 
    #开始预测
    for i in range(A.shape[0]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction
```