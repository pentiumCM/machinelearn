# 本项目主要是机器学习的算法
## 一、包说明：
+ docs：文档模块，包含数据集，训练好的模型文件
    + dataset：数据集
    + model：训练好的模型文件
    
+ machinelearn：源代码
    + application_scene：应用场景模块
    + basic：基础模块
    + data_preprocess：数据预处理
        + pca_demo.py：主成分分析（降维）
    + bayes：贝叶斯
        + naive_bayes.py：朴素贝叶斯分类器
    + cluster_analysis：聚类模块
        + density_clustering：密度聚类
        + grid_clustering：网格聚类
        + hierarchical_clustering：层次聚类
        + partition_clustering：划分聚类
    + neural_network：神经网络算法
    + keras_learn：Keras框架学习模块
    + opencv_learn：opencv框架学习模块
    + scikit_learn：sklearn框架学习模块
    + tensorflow_learn：TensorFlow框架学习
    + utils：工具模块
        + loaddata_utils.py：加载数据集工具
        + pca_utils.py：pca降维工具
        + plt_utils.py：matplotlib绘图工具
        + scaler_utils.py：数据标准化/归一化工具
        
## 二、环境搭建
1. tensorflow-gpu

| tensorflow | keras | python |
|-|-|-|
| 1.14.0| 2.2.5| 3.6|