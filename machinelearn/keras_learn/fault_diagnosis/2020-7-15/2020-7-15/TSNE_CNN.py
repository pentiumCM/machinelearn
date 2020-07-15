# 特征可视化过程
# 使用TSNE工具绘制迭代过程中CNN提取的特征数据分布情况
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKAgg")
from sklearn.manifold import TSNE
from keras_learn.fault_diagnosis import bearingPro
from sklearn.metrics import silhouette_score

import numpy as np
from keras.utils import to_categorical

import tensorflow as tf

length = 1024  # 每个样本的长度
number = 1000  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.7, 0.2, 0.1]  # 测试集验证集划分比例

path = r'F:\develop_code\python\machinelearn\machinelearn\keras_learn\fault_diagnosis\data'
train_x, train_y, valid_x, valid_y, test_x, test_y = bearingPro.prepro(d_path=path, length=length,
                                                                       number=number,
                                                                       normal=normal,
                                                                       rate=rate,
                                                                       enc=True, enc_step=28)

all_dense_data = np.load(r"feature\dense1_data.npy")
all_label = train_y
data = all_dense_data  # <7000,100>
label = all_label  # <7000,10>


# label_ = np.argmax(label)


# 画图
def plot_embedding(data, label, title):
    # “data为n * 2
    # 矩阵，label为n * 1
    # 向量，对应着data的标签, title未使用”

    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []

    for i in range(data.shape[0]):  # data.shape[0] = 7000
        if ((label[i] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all()):
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if ((label[i] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all()):
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if ((label[i] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all()):
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).all()):
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()):
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()):
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]).all()):
            type7_x.append(data[i][0])
            type7_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).all()):
            type8_x.append(data[i][0])
            type8_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).all()):
            type9_x.append(data[i][0])
            type9_y.append(data[i][1])
        if ((label[i] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).all()):
            type10_x.append(data[i][0])
            type10_y.append(data[i][1])

    # color = plt.cm.Set3(0)
    # color = np.array(color).reshape(1, 4)
    color1 = plt.cm.Set3(1)
    color1 = np.array(color1).reshape(1, 4)
    color2 = plt.cm.Set3(2)
    color2 = np.array(color2).reshape(1, 4)
    color3 = plt.cm.Set3(3)
    color3 = np.array(color3).reshape(1, 4)

    type1 = plt.scatter(type1_x, type1_y, s=10, c='r')
    type2 = plt.scatter(type2_x, type2_y, s=10, c='g')
    type3 = plt.scatter(type3_x, type3_y, s=10, c='b')
    type4 = plt.scatter(type4_x, type4_y, s=10, c='k')
    type5 = plt.scatter(type5_x, type5_y, s=10, c='c')
    type6 = plt.scatter(type6_x, type6_y, s=10, c='m')
    type7 = plt.scatter(type7_x, type7_y, s=10, c='y')
    type8 = plt.scatter(type8_x, type8_y, s=10, c=color1)
    type9 = plt.scatter(type9_x, type9_y, s=10, c=color2)
    type10 = plt.scatter(type10_x, type10_y, s=10, c=color3)
    plt.legend((type1, type2, type3, type4, type5, type6, type7, type8, type9, type10),
               ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
               loc=(0.97, 0.5))

    # plt.xticks(np.linspace(int(x_min[0]), math.ceil(x_max[0]), 5))
    # plt.yticks(np.linspace(int(x_min[1]), math.ceil(x_max[1]), 5))
    plt.xticks()
    plt.yticks()
    # plt.title(title)

    ax.spines['right'].set_visible(False)  # 去除右边框
    ax.spines['top'].set_visible(False)  # 去除上边框
    return fig


# 将特征数据进行降维
def plot_2D(data, label):
    # “data为提取的特征数据，epoch未使用”
    # n_samples, n_features = data.shape
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)  # 使用TSNE对特征降到二维
    # t0 = time()
    result = tsne.fit_transform(data)  # 降维后的数据    result.shape = <7000,2>
    # print(result.shape)
    # 画图
    fig = plot_embedding(result, label, 't-SNE embedding of the digits (time %.2fs)')
    # % (time() - t0))
    fig.subplots_adjust(right=0.8)  # 图例过大，保存figure时无法保存完全，故对此参数进行调整

    # 由one-hot转换为普通np数组
    new_label = [np.where(r==1)[0][0] for r in label]

    score = silhouette_score(data, new_label)
    print("score:", score)


plot_2D(data, label)
# 保存
plt.savefig(r"feature\TSNE_CNN.png")
plt.show()
