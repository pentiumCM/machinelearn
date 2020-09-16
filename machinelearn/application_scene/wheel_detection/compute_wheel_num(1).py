# -*- coding: utf-8 -*-
"""
全局轮胎坐标分析计算轮胎数量
@Time    : 2020-09-11 9:12
@FileName: compute_wheel_num.py
@Software: PyCharm
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle


class ComputeWheelNum1:
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


if __name__ == '__main__':
    with open("F:/develop_code/python/machinelearn/machinelearn/application_scene/wheel_detection/pickle_test",
              "rb") as f:
        data = pickle.load(f)
        print(data)
        computeWheelNum = ComputeWheelNum1(detection_coo_list=data, wheel_target=["wheel"])
        computeWheelNum.do_compute_cluster()
