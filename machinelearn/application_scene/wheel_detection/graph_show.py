# -*- coding: utf-8 -*-
"""
可视化
@Time    : 2020-09-11 17:05
@Author  : tx
@FileName: graph_show.py
@Software: PyCharm
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    all_wheel_centers = []
    with open("D:/MyProject/newcc_ai/gpu_video_recognize/expert_server/wheel_data", "rb") as f:
        data = pickle.load(f)
        for frame_index in range(len(data)):
            for detect_param in data[frame_index]:
                if detect_param["detect_label"] in ["wheel"]:
                    wheel_center_x = (detect_param["pt1"][0] + detect_param["pt2"][0]) / 2
                    all_wheel_centers.append([frame_index + 1, wheel_center_x])

    d = pd.read_csv("F:/project/tx/filter/wheel_6.csv")[["frame_index", "wheel_centers"]]
    d1 = d[d["wheel_centers"] <= 300]
    d2 = d1[d1["wheel_centers"] >= 280]
    d2["wheel_centers"]=0
    print(d2)
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()
    sns.relplot(x="frame_index", y="wheel_centers",
                data=d2)
    plt.show()

    X = np.array(d2)
    print(X)
    # X = StandardScaler().fit_transform(b)
    db = DBSCAN(eps=2, min_samples=2).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("聚类簇数{a}".format(a=n_clusters_))
    print("噪声点数{b}".format(b=n_noise_))
    # 可视化
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
