from sklearn import svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import numpy
import random

comma_tokenizer = lambda x: jieba.cut(x, cut_all=True)


# 调整了格式，一行是一条数据
def inputdata(filename):
    f = open(filename, 'r', encoding='utf-8')
    linelist = f.readlines()
    return linelist


# 按比例划分训练集与测试集
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = dataset
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return trainSet, copy


# 将问题和标签分割开来，返回words, tags
def splitset(trainset, testset):
    train_words = []
    train_tags = []
    test_words = []
    test_tags = []
    for i in trainset:
        i = i.strip()
        # index = i.index(':')
        train_words.append(i[:-2])
        # print i
        train_tags.append(int(i[-1]))

    for i in testset:
        i = i.strip()
        # index = i.index(':')
        test_words.append(i[:-2])
        # print i
        test_tags.append(int(i[-1]))

    return train_words, train_tags, test_words, test_tags


# 将训练集和测试集的问句向量化，为作为模型训练的输入做准备
def covectorize(train_words, test_words):
    v = CountVectorizer(tokenizer=comma_tokenizer, binary=False, decode_error='ignore', stop_words='english')
    train_data = v.fit_transform(train_words)
    test_data = v.transform(test_words)
    return train_data, test_data


# 创建svm分类器,返回clf  ,kenel:rbf,linear,poly
def train_clf(train_data, train_tags):
    clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                  gamma='auto', kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf.fit(train_data, numpy.asarray(train_tags))

    return clf


# 得到准确率和召回率
def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='macro')
    m_recall = metrics.recall_score(actual, pred, average='macro')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))


# 用于预测时，出现错误
def covectorize_one_sent(question):
    v = CountVectorizer(tokenizer=comma_tokenizer, binary=False, decode_error='ignore', stop_words='english')
    return v.transform(question)


if __name__ == '__main__':
    linelist = inputdata('question_type.txt')

    # 按比例划分成训练集和测试集
    trainset, testset = splitDataset(linelist, 0.7)

    # 将训练集和测试集的问句以及类型划分开来
    train_words, train_tags, test_words, test_tags = splitset(trainset, testset)

    # 将训练集和测试集向量化
    train_data, test_data = covectorize(train_words, test_words)

    # SVM分类器模型构造与训练
    clf = train_clf(train_data, train_tags)

    # pred_txt = covectorize_one_sent("剥落的原因是什么")

    test_pred = clf.predict(test_data)

    print(test_pred)
    # re = clf.predict(test_data)

    # evaluate(numpy.asarray(test_tags), re)

# 计算模型的准确度
# print("训练集准确率：",clf.score(train_x,train_y))
# print("\n")
# print("测试集准确率:",clf.score(test_x,test_y))
