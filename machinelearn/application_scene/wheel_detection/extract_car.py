#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : extract_car.py
@Time    : 2020/10/20 9:31
@desc	 : 提取一帧中的目标车  - 提取规则:图片前一半中的第一辆车
'''

import numpy as np

# car_pt_list：车的左右坐标
# car_data = [
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (445, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (445, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (445, 455), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (445, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (445, 455), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 411.0], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 411.0], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (445, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (442, 457), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 442]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 411.0], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 411.0], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (445, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 411.0], 'car_pt_list': [0, 445]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 282.0, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [282.0, 410.5], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 411.0], 'car_pt_list': [0, 443]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 411.0], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 411.0],
#      'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 411.0, 'wheel_last_coo': (444, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 411.0], 'car_pt_list': [0, 444]},
#     {'left_wheel_center_x': 281.5, 'right_wheel_center_x': 410.5, 'wheel_last_coo': (443, 456), 'car_type': 'car',
#      'wheel_center_list': [281.5, 410.5], 'car_pt_list': [0, 443]}]


car_data_v2 = [{'car_pt_list': [200, 0, 317, 529]}, {'car_pt_list': [316, 528]}, {'car_pt_list': [315, 527]},
               {'car_pt_list': [311, 528]}, {'car_pt_list': [308, 529]}, {'car_pt_list': [306, 530]},
               {'car_pt_list': [305, 527]}, {'car_pt_list': [301, 527]}, {'car_pt_list': [298, 526]},
               {'car_pt_list': [296, 525]}, {'car_pt_list': [295, 524]}, {'car_pt_list': [294, 519]},
               {'car_pt_list': [286, 527]}, {'car_pt_list': [282, 527]}, {'car_pt_list': [279, 526]},
               {'car_pt_list': [277, 525]}, {'car_pt_list': [272, 526]}, {'car_pt_list': [270, 527]},
               {'car_pt_list': [268, 527]}, {'car_pt_list': [267, 527]}, {'car_pt_list': [259, 522]},
               {'car_pt_list': [260, 521]}, {'car_pt_list': [258, 517]}, {'car_pt_list': [256, 519]},
               {'car_pt_list': [252, 517]}, {'car_pt_list': [250, 515]}, {'car_pt_list': [248, 514]},
               {'car_pt_list': [246, 513]}, {'car_pt_list': [243, 513]}, {'car_pt_list': [241, 512]},
               {'car_pt_list': [236, 512]}, {'car_pt_list': [234, 512]}, {'car_pt_list': [230, 510]},
               {'car_pt_list': [228, 511]}, {'car_pt_list': [225, 510]}, {'car_pt_list': [223, 509]},
               {'car_pt_list': [220, 508]}, {'car_pt_list': [219, 508]}, {'car_pt_list': [213, 507]},
               {'car_pt_list': [211, 506]}, {'car_pt_list': [209, 504]}, {'car_pt_list': [205, 504]},
               {'car_pt_list': [201, 503]}, {'car_pt_list': [195, 504]}, {'car_pt_list': [192, 504]},
               {'car_pt_list': [183, 500]}, {'car_pt_list': [182, 500]}, {'car_pt_list': [179, 501]},
               {'car_pt_list': [178, 500]}, {'car_pt_list': [167, 507]}, {'car_pt_list': [164, 506]},
               {'car_pt_list': [156, 505]}, {'car_pt_list': [155, 503]}, {'car_pt_list': [151, 501]},
               {'car_pt_list': [149, 501]}, {'car_pt_list': [143, 500]}, {'car_pt_list': [143, 499]},
               {'car_pt_list': [139, 498]}, {'car_pt_list': [135, 497]}, {'car_pt_list': [127, 498, 479, 561]},
               {'car_pt_list': [123, 497, 479, 562]}, {'car_pt_list': [116, 496, 478, 562]},
               {'car_pt_list': [115, 494, 478, 562]}, {'car_pt_list': [106, 497, 478, 562]},
               {'car_pt_list': [106, 492, 477, 563]}, {'car_pt_list': [103, 493, 477, 563]},
               {'car_pt_list': [102, 492, 476, 563]}, {'car_pt_list': [90, 489, 476, 564]},
               {'car_pt_list': [87, 489, 476, 564]}, {'car_pt_list': [80, 489, 469, 563]},
               {'car_pt_list': [77, 488, 468, 563]}, {'car_pt_list': [69, 485, 467, 563]},
               {'car_pt_list': [60, 485, 466, 563]}, {'car_pt_list': [53, 486, 466, 564]},
               {'car_pt_list': [45, 485, 466, 564]}, {'car_pt_list': [39, 484, 465, 565]},
               {'car_pt_list': [30, 486, 467, 563]}, {'car_pt_list': [26, 485, 466, 563]},
               {'car_pt_list': [16, 484, 467, 563]}, {'car_pt_list': [14, 484, 466, 563]},
               {'car_pt_list': [9, 487, 466, 562]}, {'car_pt_list': [10, 486, 466, 563]},
               {'car_pt_list': [4, 479, 465, 563]}, {'car_pt_list': [2, 479, 465, 563]},
               {'car_pt_list': [2, 477, 464, 563]}, {'car_pt_list': [3, 477, 464, 563]},
               {'car_pt_list': [4, 477, 463, 562]}, {'car_pt_list': [4, 473, 463, 563]},
               {'car_pt_list': [3, 473, 463, 563]}, {'car_pt_list': [5, 471, 462, 563]},
               {'car_pt_list': [5, 467, 462, 562]}, {'car_pt_list': [5, 466, 462, 563]},
               {'car_pt_list': [3, 467, 460, 562]}, {'car_pt_list': [1, 467, 460, 562]},
               {'car_pt_list': [2, 465, 459, 562]}, {'car_pt_list': [4, 463, 458, 562]},
               {'car_pt_list': [1, 465, 457, 562]}, {'car_pt_list': [0, 467, 457, 562]},
               {'car_pt_list': [0, 465, 456, 562]}, {'car_pt_list': [0, 461, 455, 561]},
               {'car_pt_list': [5, 457, 455, 561]}, {'car_pt_list': [4, 453, 454, 561]},
               {'car_pt_list': [4, 451, 453, 562]}, {'car_pt_list': [3, 449, 451, 561]},
               {'car_pt_list': [1, 451, 450, 562]}, {'car_pt_list': [5, 447, 450, 561]},
               {'car_pt_list': [7, 447, 449, 561]}, {'car_pt_list': [7, 448, 449, 560]},
               {'car_pt_list': [6, 447, 448, 559]}, {'car_pt_list': [10, 446, 447, 560]},
               {'car_pt_list': [5, 446, 446, 560]}, {'car_pt_list': [9, 443, 445, 560]},
               {'car_pt_list': [7, 445, 444, 560]}, {'car_pt_list': [0, 443, 444, 559]},
               {'car_pt_list': [2, 440, 443, 560]}, {'car_pt_list': [0, 441, 443, 559]},
               {'car_pt_list': [0, 438, 442, 559]}, {'car_pt_list': [0, 440, 441, 558]},
               {'car_pt_list': [0, 440, 442, 559]}, {'car_pt_list': [0, 436, 441, 558]},
               {'car_pt_list': [0, 436, 441, 558]}, {'car_pt_list': [0, 433, 440, 558]},
               {'car_pt_list': [0, 432, 439, 558]}, {'car_pt_list': [0, 431, 439, 558]},
               {'car_pt_list': [0, 429, 438, 557]}, {'car_pt_list': [0, 428, 438, 557]},
               {'car_pt_list': [0, 428, 438, 557]}, {'car_pt_list': [0, 426, 437, 557]},
               {'car_pt_list': [0, 426, 438, 557]}, {'car_pt_list': [0, 426, 436, 557]},
               {'car_pt_list': [0, 425, 436, 556]}, {'car_pt_list': [0, 426, 435, 556]},
               {'car_pt_list': [0, 426, 435, 556]}, {'car_pt_list': [0, 425, 435, 555]},
               {'car_pt_list': [1, 423, 434, 555]}, {'car_pt_list': [0, 423, 434, 555]},
               {'car_pt_list': [0, 422, 431, 550]}, {'car_pt_list': [1, 420, 430, 550]},
               {'car_pt_list': [1, 420, 430, 549]}, {'car_pt_list': [1, 419, 429, 549]},
               {'car_pt_list': [1, 419, 428, 550]}, {'car_pt_list': [1, 418, 428, 549]},
               {'car_pt_list': [2, 416, 428, 549]}, {'car_pt_list': [2, 417, 427, 548]},
               {'car_pt_list': [0, 416, 426, 547]}, {'car_pt_list': [0, 416, 426, 547]},
               {'car_pt_list': [3, 415, 424, 547]}, {'car_pt_list': [4, 414, 424, 547]},
               {'car_pt_list': [1, 414, 423, 546]}, {'car_pt_list': [1, 412, 422, 546]},
               {'car_pt_list': [0, 409, 421, 545]}, {'car_pt_list': [0, 408, 420, 546]},
               {'car_pt_list': [1, 403, 419, 546]}, {'car_pt_list': [2, 402, 418, 546]},
               {'car_pt_list': [2, 400, 418, 545]}, {'car_pt_list': [0, 392, 416, 543]},
               {'car_pt_list': [0, 392, 415, 543]}, {'car_pt_list': [0, 390, 408, 562]},
               {'car_pt_list': [0, 389, 407, 562]}, {'car_pt_list': [0, 386, 407, 560]},
               {'car_pt_list': [0, 385, 407, 561]}, {'car_pt_list': [0, 384, 404, 561]},
               {'car_pt_list': [0, 382, 403, 560]}, {'car_pt_list': [0, 379, 402, 559]},
               {'car_pt_list': [0, 376, 401, 557]}, {'car_pt_list': [0, 370, 400, 556]},
               {'car_pt_list': [0, 369, 399, 556]}, {'car_pt_list': [1, 361, 397, 555]},
               {'car_pt_list': [1, 358, 397, 555]}, {'car_pt_list': [3, 354, 395, 554]},
               {'car_pt_list': [2, 352, 395, 553]}, {'car_pt_list': [3, 345, 392, 554]},
               {'car_pt_list': [3, 342, 391, 554]}, {'car_pt_list': [2, 337, 391, 554]},
               {'car_pt_list': [3, 335, 390, 554]}, {'car_pt_list': [5, 326, 385, 552]},
               {'car_pt_list': [4, 326, 384, 551]}]

half_point = 320
start_point = 0
hasfind_start = False

target_car = []

for frame_car_item in car_data_v2:
    car_pt_list = frame_car_item['car_pt_list']
    car_pt_list.sort()

    # 1. 寻找目标组的起点
    # 找目标组的第一个坐标点, 即开头位置: half向下找最近的认为是目标组第一个
    if (bool(1 - hasfind_start)):
        # 最大点都小于half位置的情况,则直接把最大点位置当做开头
        if (car_pt_list[len(car_pt_list) - 1] <= half_point):
            start_point = car_pt_list[len(car_pt_list) - 1]
            hasfind_start = True
        else:
            for index in range(len(car_pt_list) - 1):
                if (car_pt_list[index] <= half_point and car_pt_list[index + 1] > half_point):
                    start_point = car_pt_list[index]
                    hasfind_start = True

    # 1.1 构建目标组的车的左右坐标
    # 找到第一个大于起点的
    for car_right_point in car_pt_list:
        if (car_right_point > start_point):
            target_car_right = car_right_point
            # target_car_item =

    # 2. 寻找目标组下一帧的坐标点:计算离下一帧最近的点 该组在上一帧目标组的下方 认为该点是目标组
    # 上下容错区间为(5,10)

print("hello")

# for frame_item in car_data:
#     car_pt_list = frame_item['car_pt_list']
#     for car_point in car_pt_list:
#         print(car_point)

print("hello")
print("hello")
print("hello")
