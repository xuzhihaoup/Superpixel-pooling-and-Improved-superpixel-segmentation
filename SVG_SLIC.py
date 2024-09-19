# -*- coding: UTF-8 -*-
# @Author  : XUZHIHAO
# @Time    : 2024/9/17 15:04
# @File    : SVG_SLIC.py
# @Function: 通过平均池化求取超像素中心，在去寻找子区域

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color, data
#计算颜色距离
def color_dis(x1,x2):
    return int((x1[0]-x2[0])**2+(x1[1]-x2[1])**2+(x1[2]-x2[2])**2)
#计算空间距离
def space_dis(y1,y2):
    return int((y1[0]-y2[0])**2+(y1[1]-y2[1])**2)


def avg_slic_center_points(image_numpy,kernel_size):
    """
    :param image_numpy: 输入图像
    :param kernel_size: 核大小
    :return: 聚类中心参数
    """
    h,w,c= image_numpy.shape
    center_points = []  # 超像素点
    for i in range(0,h,kernel_size):
        for j in range(0,w,kernel_size):
            # 对每个颜色通道分别进行求和并取平均值
            R = image_numpy[i:i + kernel_size, j:j + kernel_size, 0].mean()
            G = image_numpy[i:i + kernel_size, j:j + kernel_size, 1].mean()
            B = image_numpy[i:i + kernel_size, j:j + kernel_size, 2].mean()
            center_points.append([i+1+kernel_size//2,j+1+kernel_size//2,R,G,B])
    return center_points
def slic_find_point(image_numpy,center_points,kernel_size,control_rate):
    """
    :param image_numpy: 输入图像
    :param center_points: 聚类中心
    :param kernel_size: 核大小
    :param control_rate: 空间控制因子
    :return: 分割标签图
    """
    h,w,c= image_numpy.shape
    s = int(np.sqrt(h * w / (h * w//kernel_size^2)))
    distances_map = np.full((h, w), np.inf)  # 存储到超像素点距离
    labels = np.ones((h, w), dtype=np.int32)  # 存储标签
    for k in range(len(center_points)):
        ci, cj, cr, cg, cb = center_points[k]
        center_temp_data_DIS = [ci, cj]
        center_temp_data_RGB = [cr, cg, cb]
        for di in range(-kernel_size, kernel_size):
            for dj in range(-kernel_size, kernel_size):
                ni = int(ci + di)
                nj = int(cj + dj)
                # 确保像素位置合法
                if ni < 0 or ni >= h or nj < 0 or nj >= w:
                    # print(f"坐标非法在({ni},{nj})")
                    continue
                other_temp_data_RGB = [image_numpy[ni, cj, 0], image_numpy[ni, nj, 1], image_numpy[ni, nj, 2]]
                other_temp_data_DIS = [ni, nj]
                Dc = color_dis(center_temp_data_RGB, other_temp_data_RGB)
                Ds = space_dis(center_temp_data_DIS, other_temp_data_DIS)
                D = np.sqrt(Dc + Ds / s * control_rate)
                # 更新距离地图 以及标签
                if D < distances_map[ni, nj]:
                    distances_map[ni, nj] = D
                    labels[ni, nj] = k
    return labels
def show_slic_image(image_np,labels_slic):
    boundaries = np.zeros_like(image_np)
    boundaries = image_np
    h, w = labels_slic.shape
    for i in range(5, h - 5):
        for j in range(5, w - 5):
            if labels_slic[i, j] != labels_slic[i, j + 1] and labels_slic[i, j] == labels_slic[i, j - 5] \
                or labels_slic[i, j] != labels_slic[i + 1, j] and labels_slic[i, j] == labels_slic[i - 5, j]:
                boundaries[i, j] = [255, 255, 0]  # 用红色标记边界
                boundaries[i, j + 1] = [255, 255, 0]  # 用红色标记边界
                # boundaries[i, j - 1] = [255, 255, 255]  # 用红色标记边界
            else:
                pass
    plt.imsave('boundaries.png', boundaries)
    plt.imshow(boundaries)
    plt.axis('off')
    plt.show()






image  = Image.open('ndrishtiGS_085.png').convert('RGB')
image = color.rgb2lab(image)
kernel_num = 32
image_np = np.array(image)
r,g,b = image_np[0,0]
print(image_np.shape)
center_points = avg_slic_center_points(image_np,kernel_num)
print(len(center_points))
labels = slic_find_point(image_np,center_points,kernel_num,0.5)
image_np = color.lab2rgb(image_np)
image_np = (image_np * 255).astype(np.uint8)
show_slic_image(image_np,labels)