#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : group_by_cluster.py
@Time    : 2020/9/14 10:01
@desc	 : 通过聚类的方式 实现车轮个数检测
'''

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from application_scene.wheel_detection.wheel_parms import wheel_parms_2
from application_scene.wheel_detection.wheel_parms import wheel_parms_6


class ComputeWheelNum:
    def __init__(self, detection_coo_list, wheel_target, analysis_interval=None, judge_threshold=5):
        """
        :param detection_coo_list:所有图片检测参数
        :param wheel_target: 轮子标签
        :param analysis_interval: 轮子中心点x坐标分析区间
        :param judge_threshold: 判断是否为同一个轮子中心点x偏移阈值
        """
        if analysis_interval is None:
            analysis_interval = [200, 300]
        self.detection_coo_list = detection_coo_list
        self.analysis_interval = analysis_interval
        self.wheel_target = wheel_target
        self.judge_threshold = judge_threshold

    def do_analyse(self, one_frame_param):
        wheel_centers = []
        for detect_param in one_frame_param:
            if detect_param["detect_label"] in self.wheel_target:
                wheel_center_x = (detect_param["pt1"][0] + detect_param["pt2"][0]) / 2
                if self.analysis_interval[0] <= wheel_center_x <= self.analysis_interval[1]:
                    wheel_centers.append(wheel_center_x)
        wheel_centers.sort(reverse=False)
        return wheel_centers

    def do_compute_cluster(self, cluster_radius=2, cluster_num=2):
        """
        通过对区间截取数据聚类统计出轮子数量
        使用DBSCN测试聚类效果
        :param cluster_radius:聚类区域半径
        :param cluster_num:聚类区域至少含有点数
        :return:聚类区域数量（轮胎数）
        """
        all_wheel_centers = []
        for frame_index in range(len(self.detection_coo_list)):
            one_frame_wheel_centers = self.do_analyse(self.detection_coo_list[frame_index])
            if one_frame_wheel_centers:
                for one_frame_wheel_center in one_frame_wheel_centers:
                    all_wheel_centers.append([frame_index + 1, 0])

        if all_wheel_centers:
            # X = StandardScaler().fit_transform(all_wheel_centers)
            X = all_wheel_centers
            db = DBSCAN(eps=cluster_radius, min_samples=cluster_num).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            print("聚类簇数{a}".format(a=n_clusters_))
            print("噪声点数{b}".format(b=n_noise_))
            return n_clusters_
        else:
            return None

    # 车轮检测参数
    wheel_parms = wheel_parms_6

    wheel_list = []

    for frame in range(len(wheel_parms)):
        # 获取当前的帧索引
        frame_index = frame
        # 获取当前帧中的检测的车轮参数
        wheel_center_list = wheel_parms[frame]['wheel_center_list']

        list.sort(wheel_center_list)

        # 构造 【frame, wheel_center_x】
        # 构造 【frame, 0】- 相当于投影 X 轴上
        for item in range(len(wheel_center_list)):
            wheel_center_x = wheel_center_list[item]
            # 在子空间上进行聚类
            if (wheel_center_x >= 200 and wheel_center_x <= 300):
                # wheel_list_item = np.array([frame_index, wheel_center_list[item]])
                wheel_list_item = np.array([frame_index, 0])
                wheel_list.append(wheel_list_item)

    # 2、对数据进行预处理
    # 2.1、 对坐标点进行标准化
    wheel_std_data = StandardScaler().fit_transform(wheel_list)
    # wheel_std_data = wheel_list

    # 基于密度聚类
    db = DBSCAN(eps=2, min_samples=2).fit(wheel_std_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("聚类簇数{a}".format(a=n_clusters_))
    print("噪声点数{b}".format(b=n_noise_))

    # 可视化
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = wheel_std_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = wheel_std_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':
    data = wheel_parms_2

    print(data)
    computeWheelNum = ComputeWheelNum(detection_coo_list=data, wheel_target=["wheel"])
    computeWheelNum.do_compute_cluster()
