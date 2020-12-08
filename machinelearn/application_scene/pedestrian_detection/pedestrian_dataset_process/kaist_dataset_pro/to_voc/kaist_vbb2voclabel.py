#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : seq2voc.py
@Time    : 2020/9/7 21:44
@desc	 : KAIST 数据集 VBB 标注文件转化为XML文件
'''

import os, glob
import cv2
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify


def seq2img(annos, seq_file, outdir, cam_id):
    """
    可视化标注的矩形框
    :param annos:
    :param seq_file:
    :param outdir:
    :param cam_id:
    :return:
    """
    cap = cv2.VideoCapture(seq_file)
    index = 1
    # captured frame list
    v_id = os.path.splitext(os.path.basename(seq_file))[0]
    cap_frames_index = np.sort([int(os.path.splitext(id)[0].split("_")[2]) for id in annos.keys()])
    while True:
        ret, frame = cap.read()
        print(ret)
        if ret:
            if not index in cap_frames_index:
                index += 1
                continue
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outname = os.path.join(outdir, str(cam_id) + "_" + v_id + "_" + str(index) + ".jpg")
            print("Current frame: ", v_id, str(index))
            cv2.imwrite(outname, frame)
            height, width, _ = frame.shape
        else:
            break
        index += 1
    img_size = (width, height)
    return img_size


def vbb_anno2dict(vbb_file, cam_id):
    """
    返回一个vbb文件中所有的帧的标注结果
    :param vbb_file: vbb文件
    :param cam_id:
    :return:
    """
    # 通过os.path.basename获得路径的最后部分“文件名.扩展名”
    # 通过os.path.splitext获得文件名
    filename = os.path.splitext(os.path.basename(vbb_file))[0]

    # 定义字典对象annos
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    # object info in each frame: id, pos, occlusion, lock, posv
    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]  # 可查看所有类别

    # person index
    # person_index_list = np.where(np.array(objLbl) == "person")[0]  # 只选取类别为‘person’的xml

    for frame_id, obj in enumerate(objLists):
        # frame_name = str(cam_id) + "_" + str(filename) + "_" + str(frame_id + 1) + ".jpg"

        # frame_name = '/'.join([cam_id, filename, 'I{:05d}'.format(frame_id)])
        frame_name = '_'.join([cam_id, filename, 'I{:05d}'.format(frame_id)])

        annos[frame_name] = defaultdict(list)
        annos[frame_name]["id"] = frame_name

        if len(obj[0]) > 0:
            for id, pos, occl, lock, posv in zip(
                    obj['id'][0], obj['pos'][0],
                    obj['occl'][0], obj['lock'][0],
                    obj['posv'][0]):
                id = int(id[0][0]) - 1  # for matlab start from 1 not 0

                # if not id in person_index_list:  # only use bbox whose label is person
                #     continue

                pos = pos[0].tolist()
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                annos[frame_name]["label"].append(objLbl[id])
                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)

        if not annos[frame_name]["bbox"]:
            del annos[frame_name]
    print(annos)
    return annos


def instance2xml_base(anno, img_size, bbox_type='xyxy'):
    """

    :param anno:
    :param img_size:
    :param bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)
    :return:
    """
    assert bbox_type in ['xyxy', 'xywh']
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/person'),
        E.filename(anno['id']),
        E.source(
            E.database('KAIST pedestrian'),
            E.annotation('KAIST pedestrian'),
            E.image('KAIST pedestrian'),
            E.url('https://soonminhwang.github.io/rgbt-ped-detection/')
        ),
        E.size(
            E.width(img_size[0]),
            E.height(img_size[1]),
            E.depth(4)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':

            xmin, ymin, xmax, ymax = bbox
        else:
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
        E = objectify.ElementMaker(annotate=False)
        anno_tree.append(
            E.object(
                E.name(anno['label'][index]),
                E.bndbox(
                    E.xmin(xmin),
                    E.ymin(ymin),
                    E.xmax(xmax),
                    E.ymax(ymax)
                ),

                E.pose('unknown'),
                E.truncated(0),
                E.difficult(0),
                E.occlusion(anno["occlusion"][index])
            )
        )
    return anno_tree


def parse_anno_file(vbb_inputdir, vbb_outputdir, IMAGE_SIZE):
    """

    :param vbb_inputdir:
    :param vbb_outputdir:
    :param IMAGE_SIZE: 数据集图片尺寸
    :return:
    """
    # annotation sub-directories in hda annotation input directory
    assert os.path.exists(vbb_inputdir)
    sub_dirs = os.listdir(vbb_inputdir)  # 对应set00,set01...
    for sub_dir in sub_dirs:
        print("Parsing annotations of camera: ", sub_dir)
        cam_id = sub_dir
        # 获取某一个子set下面的所有vbb文件
        vbb_files = glob.glob(os.path.join(vbb_inputdir, sub_dir, "*.vbb"))
        for vbb_file in vbb_files:
            # 返回一个vbb文件中所有的帧的标注结果
            annos = vbb_anno2dict(vbb_file, cam_id)

            if annos:
                # 组成xml文件的存储文件夹，形如“/Users/chenguanghao/Desktop/Caltech/xmlresult/”
                # vbb_outdir = os.path.join(vbb_outputdir, sub_dir, os.path.basename(vbb_file).split('.')[0])
                vbb_outdir = vbb_outputdir

                # 如果不存在
                if not os.path.exists(vbb_outdir):
                    os.makedirs(vbb_outdir)

                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    if "bbox" in anno:
                        anno_tree = instance2xml_base(anno, IMAGE_SIZE, bbox_type='xywh')
                        outfile = os.path.join(vbb_outdir, os.path.splitext(filename)[0] + ".xml")
                        print("Generating annotation xml file of picture: ", filename)
                        # 生成最终的xml文件，对应一张图片
                        etree.ElementTree(anno_tree).write(outfile, pretty_print=True)


if __name__ == "__main__":
    """
    vbb标注文件 转 VOC标注 的 xml 入口函数
    :return:
    """

    # 数据集图片尺寸
    IMAGE_SIZE = (640, 512)  # KAIST Multispectral Benchmark

    label_path = 'F:/experiment/data/KAIST/annotations'

    vbb_inputdir = os.path.join(label_path, 'annotations-vbb')
    vbb_outputdir = os.path.join(label_path, 'annotations-xml')
    parse_anno_file(vbb_inputdir, vbb_outputdir)
