import numpy as np
import open3d
import sys
import os
import math
import cv2

def read_calib_file(calib_path):
    out = dict()
    for line in open(calib_path, 'r'):
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        val = line.split(':')
        key_name = val[0].strip()
        val = np.asarray(val[-1].strip().split(' '), dtype='f8')
        if len(val) == 12:
            out[key_name] = val.reshape(3, 4)
        elif len(val) == 9:
            out[key_name] = val.reshape(3, 3)

    return out


def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# ----------------------------- Read parameters from predict label folder and generate necessary 3D box?? ---------------------
# Index should be used as reference
# for each detection output files in result folder
def read_label(label_path,index):

    eval_dics = {}
    eval_dics [index] = []
    #Bbox_dicts = {}

    f = open(label_path, 'r')
    lines = []
    for line in f.readlines():
        lines.append(line)



    for line in lines:  ####for each object in each txt file

        Bbox_dict =  {}

        bbox2des = []
        bbox2des_center = []
        centers_list = []
        size_list = []
        rotation_y_list = []
        parameter_list = line.strip().split(" ")

        class_id = parameter_list[0]
        xmin = np.asarray(parameter_list[4] ,dtype = 'float32') # xmin
        ymin = np.asarray(parameter_list[5] ,dtype = 'float32')  # ymin
        xmax = np.asarray(parameter_list[6] ,dtype = 'float32')  # xmax
        ymax = np.asarray(parameter_list[7] ,dtype = 'float32') # ymax
        box2d = np.array([xmin, ymin, xmax, ymax])
        #bbox2des.append(box2d)
        box2d_center = np.array([(xmax - xmin) / 2, (ymax - ymin) / 2])
        #bbox2des_center.append(box2d_center)

        height = parameter_list[8]
        width = parameter_list[9]
        length = parameter_list[10]

        size = np.array([height, width, length])
        #size_list.append(size)  ## h,l,w of Bbox

        tx = float(parameter_list[11])
        ty = float(parameter_list[12])
        tz = float(parameter_list[13])

        center_list = np.array([tx, ty, tz])  # need to rotation along y    -rotation_y
        #centers_list.append(center_list)  # 3d center in camera coordinates??????
        rotation_y = parameter_list[-2]
        score = parameter_list[-1]
        #rotation_y_list.append(rotation_y)

        # predict_3d_center= provider.rotate_pc_along_y(,    -rotation_y)
        Bbox_dict["center"] = center_list
        Bbox_dict["height"] = float(height)
        Bbox_dict["width"] = float(width)
        Bbox_dict["length"] = float(length)
        Bbox_dict["r_y"] = wrapToPi(float(rotation_y))



        #Bbox_dicts.update(Bbox_dict)

        eval_dics[index].append(Bbox_dict)

    return eval_dics

def read_gt(gt_path, sequence):
    gt_dics = {}
    gt_dics[sequence] = []
    f = open(gt_path, 'r')
    lines = []
    for line in f.readlines():
        lines.append(line)

    for line in lines:
        gt_bbox_dics = {}
        parameter_list = line.strip().split(" ")
        xmin = np.asarray(parameter_list[4], dtype='float32')  # xmin
        ymin = np.asarray(parameter_list[5], dtype='float32')  # ymin
        xmax = np.asarray(parameter_list[6], dtype='float32')  # xmax
        ymax = np.asarray(parameter_list[7], dtype='float32')  # ymax


        gt_bbox_dics["xmin"] = xmin
        gt_bbox_dics["ymin"] = ymin
        gt_bbox_dics["xmax"] = xmax
        gt_bbox_dics["ymax"] = ymax

        #box2d = np.array([xmin, ymin, xmax, ymax])


        gt_dics[sequence].append(gt_bbox_dics)

    return gt_dics



