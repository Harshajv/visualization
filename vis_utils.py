import numpy as np
import os
import sys
import open3d
import math
import cv2
import pickle
sys.path.append("/home/kanging/Documents/Masterarbeit/Open3D/build/lib")
sys.path.append("/home/kangning/Documents/Masterarbeit/frustum-pointnets/train/")

pickle_path= "/home/kangning/Documents/Masterarbeit/frustum-pointnets/train/detection_results_v1.pickle"
from image_generator import ScaleCols, ScaleRows, ScaleY



########Create 3D Bounding Box#################################################################################################################

def create3Dbbox(center,h,w,l,r_y,type="Predict"):               ## For Point Cloud, IN Open3d, to draw a cubic needs {8 Points, 12 Lines, Colors}

    if type == "Predict":
        color = [1,0.75,0]    # Normalized RGB
        front_color = [1,0,0]# Normalized RGB
        #color1=
    else : #(if type == "gt": )
        color = [1,0,0.75]    # Normalized RGB 
        front_color = [0,0.9,1] # Normalized RGB

    R_Matrix_y = np.asarray([[math.cos(r_y),0,math.sin(r_y)],[0,1,0],[-math.sin(r_y),0,math.cos(r_y)]],dtype= np.float64)

    R_Matrix_y_rect = np.asarray([[math.cos(r_y + np.pi/2),0,math.sin(r_y + np.pi/2)],[0,1,0],[-math.sin(r_y + np.pi/2),0,math.cos(r_y + np.pi/2)]],dtype= np.float64)

    R_Matrix_x_rect = np.asarray([[1,0,0],[0,math.cos(np.pi/2),math.sin(np.pi/2)],[0,-math.sin(np.pi/2),math.cos(np.pi/2)]], dtype = np.float64)
    

    p0 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,0,w/2.0],dtype = 'float32').flatten())
    p1 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,0,w/2.0],dtype = 'float32').flatten())
    p2 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,0,-w/2.0],dtype = 'float32').flatten())
    p3 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,0,-w/2.0],dtype = 'float32').flatten())  
    p4 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,-h,w/2.0],dtype = 'float32').flatten())
    p5 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,-h,w/2.0],dtype = 'float32').flatten())
    p6 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,-h,-w/2.0],dtype = 'float32').flatten())
    p7 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,-h,-w/2.0],dtype = 'float32').flatten())


    p0_3 = center + np.dot(R_Matrix_y, np.asarray([l/2.0 ,0, 0], dtype= 'float32').flatten())
    p1_2 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0 ,0, 0], dtype= 'float32').flatten())
    p4_7 = center + np.dot(R_Matrix_y, np.asarray([l/2.0 ,-h, 0], dtype= 'float32').flatten())
    p5_6 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0 ,-h, 0], dtype= 'float32').flatten())
    p0_1 = center + np.dot(R_Matrix_y, np.asarray([0,0,w/2.0], dtype= 'float32').flatten())
    p3_2 = center + np.dot(R_Matrix_y, np.asarray([0,0,-w/2.0], dtype= 'float32').flatten())
    p4_5 = center + np.dot(R_Matrix_y, np.asarray([0,-h,w/2.0], dtype= 'float32').flatten())
    p7_6 = center + np.dot(R_Matrix_y, np.asarray([0,-h,-w/2.0], dtype= 'float32').flatten())
    p0_4 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,-h/2.0,w/2.0], dtype= 'float32').flatten())
    p3_7 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,-h/2.0,-w/2.0], dtype= 'float32').flatten())
    p1_5 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,-h/2.0,w/2.0], dtype= 'float32').flatten())
    p2_6 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,-h/2.0,-w/2.0], dtype= 'float32').flatten())

    p0_1_3_2 = center
    #print(p0_1_3_2)
    #print(p0_1_3_2)


    length_0_3 = np.linalg.norm(p0-p3)
    cylinder_0_3 = open3d.create_mesh_cylinder (radius = 0.025,height = length_0_3)
    cylinder_0_3.compute_vertex_normals()
    transform_0_3 = np.eye(4)
    transform_0_3[0:3, 0:3] = R_Matrix_y
    transform_0_3[0:3 , 3] = p0_3
    cylinder_0_3.transform(transform_0_3)
    cylinder_0_3.paint_uniform_color(front_color)

    length_1_2 = np.linalg.norm(p1-p2)
    cylinder_1_2 = open3d.create_mesh_cylinder (radius = 0.025,height = length_1_2)
    cylinder_1_2.compute_vertex_normals()
    transform_1_2 = np.eye(4)
    transform_1_2[0:3, 0:3] = R_Matrix_y
    transform_1_2[0:3 , 3] = p1_2
    cylinder_1_2.transform(transform_1_2)
    cylinder_1_2.paint_uniform_color(front_color)

    length_4_7 = np.linalg.norm(p4-p7)
    cylinder_4_7 = open3d.create_mesh_cylinder (radius = 0.025,height = length_4_7)
    cylinder_4_7.compute_vertex_normals()
    transform_4_7 = np.eye(4)
    transform_4_7[0:3, 0:3] = R_Matrix_y
    transform_4_7[0:3 , 3] = p4_7
    cylinder_4_7.transform(transform_4_7)
    cylinder_4_7.paint_uniform_color(front_color)

    length_5_6 = np.linalg.norm(p5-p6)
    cylinder_5_6 = open3d.create_mesh_cylinder (radius = 0.025,height = length_5_6)
    cylinder_5_6.compute_vertex_normals()
    transform_5_6 = np.eye(4)
    transform_5_6[0:3, 0:3] = R_Matrix_y
    transform_5_6[0:3 , 3] = p5_6
    cylinder_5_6.transform(transform_5_6)
    cylinder_5_6.paint_uniform_color(front_color)

    length_0_1 = np.linalg.norm(p0-p1)
    cylinder_0_1 = open3d.create_mesh_cylinder (radius = 0.025,height = length_0_1)
    cylinder_0_1.compute_vertex_normals()
    transform_0_1 = np.eye(4)
    transform_0_1[0:3, 0:3] = R_Matrix_y_rect
    transform_0_1[0:3 , 3] = p0_1
    cylinder_0_1.transform(transform_0_1)
    cylinder_0_1.paint_uniform_color(front_color)
    
    length_3_2 = np.linalg.norm(p3-p2)
    cylinder_3_2 = open3d.create_mesh_cylinder (radius = 0.025,height = length_3_2)
    cylinder_3_2.compute_vertex_normals()
    transform_3_2 = np.eye(4)
    transform_3_2[0:3, 0:3] = R_Matrix_y_rect
    transform_3_2[0:3 , 3] = p3_2
    cylinder_3_2.transform(transform_3_2)
    cylinder_3_2.paint_uniform_color(front_color)
    
    length_4_5 = np.linalg.norm(p4-p5)
    cylinder_4_5 = open3d.create_mesh_cylinder (radius = 0.025,height = length_4_5)
    cylinder_4_5.compute_vertex_normals()
    transform_4_5 = np.eye(4)
    transform_4_5[0:3, 0:3] = R_Matrix_y_rect
    transform_4_5[0:3 , 3] = p4_5
    cylinder_4_5.transform(transform_4_5)
    cylinder_4_5.paint_uniform_color(front_color)



    length_7_6 = np.linalg.norm(p7-p6)
    cylinder_7_6 = open3d.create_mesh_cylinder (radius = 0.025,height = length_7_6)
    cylinder_7_6.compute_vertex_normals()
    transform_7_6 = np.eye(4)
    transform_7_6[0:3, 0:3] = R_Matrix_y_rect
    transform_7_6[0:3 , 3] = p7_6
    cylinder_7_6.transform(transform_7_6)
    cylinder_7_6.paint_uniform_color(front_color)


    length_0_4 = np.linalg.norm(p0-p4)
    cylinder_0_4 = open3d.create_mesh_cylinder (radius = 0.025,height = length_0_4)
    cylinder_0_4.compute_vertex_normals()
    transform_0_4 = np.eye(4)
    transform_0_4[0:3, 0:3] = np.dot(R_Matrix_y,R_Matrix_x_rect)
    transform_0_4[0:3 , 3] = p0_4
    cylinder_0_4.transform(transform_0_4)
    cylinder_0_4.paint_uniform_color(front_color)
    
 
    length_3_7 = np.linalg.norm(p3-p7)
    cylinder_3_7 = open3d.create_mesh_cylinder (radius = 0.025,height = length_3_7)
    cylinder_3_7.compute_vertex_normals()
    transform_3_7 = np.eye(4)
    transform_3_7[0:3, 0:3] = np.dot(R_Matrix_y,R_Matrix_x_rect)
    transform_3_7[0:3 , 3] = p3_7
    cylinder_3_7.transform(transform_3_7)
    cylinder_3_7.paint_uniform_color(front_color)


    length_1_5 = np.linalg.norm(p1-p5)
    cylinder_1_5 = open3d.create_mesh_cylinder (radius = 0.025,height = length_1_5)
    cylinder_1_5.compute_vertex_normals()
    transform_1_5 = np.eye(4)
    transform_1_5[0:3, 0:3] = np.dot(R_Matrix_y,R_Matrix_x_rect)
    transform_1_5[0:3 , 3] = p1_5
    cylinder_1_5.transform(transform_1_5)
    cylinder_1_5.paint_uniform_color(front_color)

    length_2_6 = np.linalg.norm(p2-p6)
    cylinder_2_6 = open3d.create_mesh_cylinder (radius = 0.025,height = length_2_6)
    cylinder_2_6.compute_vertex_normals()
    transform_2_6 = np.eye(4)
    transform_2_6[0:3, 0:3] = np.dot(R_Matrix_y,R_Matrix_x_rect)
    transform_2_6[0:3 , 3] = p2_6
    cylinder_2_6.transform(transform_2_6)
    cylinder_2_6.paint_uniform_color(front_color)

    length_0_1_3_2 = np.linalg.norm(p0_1 - p3_2)
    cylinder_0_1_3_2 = open3d.create_mesh_cylinder(radius=0.025, height=length_0_1_3_2)
    cylinder_0_1_3_2.compute_vertex_normals()
    transform_0_1_3_2 = np.eye(4)
    transform_0_1_3_2[0:3, 0:3] = R_Matrix_y
    transform_0_1_3_2[0:3, 3] = p0_1_3_2
    cylinder_0_1_3_2.transform(transform_0_1_3_2)
    cylinder_0_1_3_2.paint_uniform_color(color)
    
    #return [cylinder_0_1_3_2, cylinder_0_3, cylinder_1_2, cylinder_4_7, cylinder_5_6, cylinder_0_1, cylinder_3_2, cylinder_4_5, cylinder_7_6, cylinder_0_4, cylinder_3_7, cylinder_1_5, cylinder_2_6]
    return [cylinder_0_3, cylinder_1_2, cylinder_4_7, cylinder_5_6, cylinder_0_1, cylinder_3_2, cylinder_4_5, cylinder_7_6, cylinder_0_4, cylinder_3_7, cylinder_1_5, cylinder_2_6]


def create3dBbox_images(center, h, w, l, calib_matrix,r_y):


        
    color = [1,0.75,0]    # Normalized RGB
    front_color = [1,0,0] # Normalized RGB

    Bbox_image = {}
    R_Matrix_y = np.asarray([[math.cos(r_y),0,math.sin(r_y)],
                          [0,1,0],
                          [-math.sin(r_y),0,math.cos(r_y)]],
                          dtype= 'float32')



    p0 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,0,w/2.0],dtype = 'float32').flatten())
    p1 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,0,w/2.0],dtype = 'float32').flatten())
    p2 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,0,-w/2.0],dtype = 'float32').flatten())
    p3 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,0,-w/2.0],dtype = 'float32').flatten())  
    p4 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,-h,w/2.0],dtype = 'float32').flatten())
    p5 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,-h,w/2.0],dtype = 'float32').flatten())
    p6 = center + np.dot(R_Matrix_y, np.asarray([-l/2.0,-h,-w/2.0],dtype = 'float32').flatten())
    p7 = center + np.dot(R_Matrix_y, np.asarray([l/2.0,-h,-w/2.0],dtype = 'float32').flatten())



    

    Bbox_image['points'] = np.asarray([p0,p1,p2,p3,p4,p5,p6,p7])
    Bbox_image['lines'] = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[3,7],[2,6]]
    Bbox_image['colors'] = [front_color]
    Bbox_image['calib_matrix'] = calib_matrix


    return Bbox_image

def scale_3d_lidar_points_into_2d_lidar_image(point):

    point_x = ScaleCols(point[2])
    point_y = ScaleRows(point[0])

    point_in_2d_lidar_image = (point_x, point_y)
    return point_in_2d_lidar_image

def  create3dBbox_on_lidar_images(center,h,w,l,r_y):
    color = [1, 0.75, 0]  # Normalized RGB
    front_color = [1, 0, 0]  # Normalized RGB

    Bbox_image = {}
    R_Matrix_y = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                             [0, 1, 0],
                             [-math.sin(r_y), 0, math.cos(r_y)]],
                            dtype='float32')

    p0 = center + np.dot(R_Matrix_y, np.asarray([l / 2.0, 0, w / 2.0], dtype='float32').flatten())
    p1 = center + np.dot(R_Matrix_y, np.asarray([-l / 2.0, 0, w / 2.0], dtype='float32').flatten())
    p2 = center + np.dot(R_Matrix_y, np.asarray([-l / 2.0, 0, -w / 2.0], dtype='float32').flatten())
    p3 = center + np.dot(R_Matrix_y, np.asarray([l / 2.0, 0, -w / 2.0], dtype='float32').flatten())
    p4 = center + np.dot(R_Matrix_y, np.asarray([l / 2.0, -h, w / 2.0], dtype='float32').flatten())
    p5 = center + np.dot(R_Matrix_y, np.asarray([-l / 2.0, -h, w / 2.0], dtype='float32').flatten())
    p6 = center + np.dot(R_Matrix_y, np.asarray([-l / 2.0, -h, -w / 2.0], dtype='float32').flatten())
    p7 = center + np.dot(R_Matrix_y, np.asarray([l / 2.0, -h, -w / 2.0], dtype='float32').flatten())

    p0_new = scale_3d_lidar_points_into_2d_lidar_image(p0)
    p1_new = scale_3d_lidar_points_into_2d_lidar_image(p1)
    p2_new = scale_3d_lidar_points_into_2d_lidar_image(p2)
    p3_new = scale_3d_lidar_points_into_2d_lidar_image(p3)
    p4_new = scale_3d_lidar_points_into_2d_lidar_image(p4)
    p5_new = scale_3d_lidar_points_into_2d_lidar_image(p5)
    p6_new = scale_3d_lidar_points_into_2d_lidar_image(p6)
    p7_new = scale_3d_lidar_points_into_2d_lidar_image(p7)

    Bbox_image['points'] = np.asarray([p0_new, p1_new, p2_new, p3_new, p4_new, p5_new, p6_new, p7_new])
    Bbox_image['lines'] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [3, 7],
                           [2, 6]]


    Bbox_image['colors'] = [front_color]


    return Bbox_image

def create_gt_2d_bbox(xmin,ymin,xmax,ymax):

    gt_2d_bbox_on_image = {}

    gt_2d_bbox_on_image["gt_2d"] = np.array([[xmin,ymin],[xmax,ymin],[xmin,ymax],[xmax,ymax]],dtype = 'int32')
    gt_2d_bbox_on_image["lines"] = [[0,1],[1,3],[3,2],[2,0]]

    return gt_2d_bbox_on_image






 





























