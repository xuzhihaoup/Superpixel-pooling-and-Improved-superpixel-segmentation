# -*- coding: utf-8 -*-
# @Author  : XUZHIHAO
# @Time    : 2024/9/5  19:44
# @Software: PyCharm
# @File    : user_test.py
# @Function:slic method
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

from skimage import color


#计算颜色距离
def color_dis(x1,x2):
    return int((x1[0]-x2[0])**2+(x1[1]-x2[1])**2+(x1[2]-x2[2])**2)
#计算空间距离
def space_dis(y1,y2):
    return int((y1[0]-y2[0])**2+(y1[1]-y2[1])**2)

def slic_method(image_data,segments_num=100,epoch_max=10,control_rate=10):
    """
    :param image_data:图像数组
    :param segments_num:分割数
    :param epoch_max:最大迭代次数
    :param control_rate:空间距离控制率
    :return:返回标签图
    """
    h,w,c = image_data.shape
    center_points = [] #超像素点
    labels =  np.ones((h, w), dtype=np.int32)#存储标签
    distances_map = np.full((h, w), np.inf)  # 存储到超像素点距离
    other_temp_data_RGB = []   #中心点四周RGB临时点
    center_temp_data_RGB = []  #中心超像素RGB临时点
    other_temp_data_DIS = []   #中心点四周坐标临时点
    center_temp_data_DIS = []  #中心超像素坐标临时点
    #计算步长
    s = int(np.sqrt(h*w/segments_num))
    for i in range(s//2,h,s):
        for j in range(s//2,w,s):
            center_points.append([i,j,image_data[i,j,0],image_data[i,j,1],image_data[i,j,2]])
    centers = np.array(center_points)
    print(centers.shape)
    centers_num = centers.shape[0]
    #开始迭代
    for i in range(epoch_max):
        print(f"第{i+1}/{epoch_max}轮迭代")
        for k in range(centers_num):
            ci,cj,cr,cg,cb = centers[k]
            center_temp_data_DIS = [ci,cj]
            center_temp_data_RGB = [cr, cg, cb]
            for di in range(-s, s):
                for dj in range(-s,s):
                   ni = int(ci+di)
                   nj = int(cj+dj)
                   # 确保像素位置合法
                   if ni < 0 or ni >= h or nj < 0 or nj >= w:
                       continue
                   other_temp_data_RGB  = [image_data[ni,nj,0],image_data[ni,nj,1],image_data[ni,nj,2]]
                   other_temp_data_DIS  = [ni,nj]
                   Dc = color_dis(center_temp_data_RGB,other_temp_data_RGB)
                   Ds = space_dis(center_temp_data_DIS,other_temp_data_DIS)
                   D = np.sqrt(Dc+Ds/s*control_rate)
                   #更新距离地图 以及标签
                   if D < distances_map[ni, nj]:
                       distances_map[ni, nj] = D
                       labels[ni, nj] = k

        # 更新每个超像素的中心
        new_centers = np.zeros_like(centers)
        #存储每个区域有多少个像素点
        class_counts = np.zeros(centers_num)
        for i_x in range(h):
            for j_y in range(w):
               k = labels[i_x,j_y] # centers[x,y,R,G,B]
               new_centers[k,0] += i_x  #同一个class x坐标求和
               new_centers[k,1] += j_y #同一个class y坐标求和
               new_centers[k,2] += int(image_data[i_x,j_y,0])#像素值求和
               new_centers[k,3] += int(image_data[i_x, j_y, 1])  # 像素值求和
               new_centers[k,4] += int(image_data[i_x, j_y, 2])  # 像素值求和
               class_counts[k] += 1
        new_centers = new_centers.astype(np.float64)
        for i_k in range(centers_num):
            if class_counts[i_k] > 0:
                new_centers[i_k] /= float(class_counts[i_k])
        #更新后的超像素中心  更新方式类似k_mean 聚类
        centers = new_centers.astype(int)
    return labels , centers
def show_slic_pooling(slic_centers):
    boundaries = np.zeros((int(math.sqrt(slic_centers.shape[0])),int(math.sqrt(slic_centers.shape[0])),3))
    h,w= boundaries.shape[0:2]
    point_index = 0
    for i in range(h):
       for j in range(w):
           boundaries[i,j] = [slic_centers[point_index,2],slic_centers[point_index,3],slic_centers[point_index,4]]
           point_index += 1
    print("图片打印")
    # plt.imshow(boundaries/255)
    # plt.axis('off')
    # plt.show()
    # 分别提取 R、G、B 通道
    R = boundaries[:, :, 0]  # 红色通道
    G = boundaries[:, :, 1]  # 绿色通道
    B = boundaries[:, :, 2]  # 蓝色通道

    # 创建图形并显示每个通道
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    ax[0].imshow(R, cmap='Reds')  # 显示红色通道
    ax[0].set_title('Red Channel')
    ax[0].axis('off')

    ax[1].imshow(G, cmap='Greens')  # 显示绿色通道
    ax[1].set_title('Green Channel')
    ax[1].axis('off')

    ax[2].imshow(B, cmap='Blues')  # 显示蓝色通道
    ax[2].set_title('Blue Channel')
    ax[2].axis('off')

    ax[3].imshow(boundaries/255, cmap='Blues')  # 显示所有通道
    ax[3].set_title('All Channel')
    ax[3].axis('off')
    plt.show()
def show_slic_image(image,lanels_slic):
    boundaries = np.zeros_like(image)
    boundaries = image
    h, w = lanels_slic.shape
    for i in range(5, h - 5):
        for j in range(5, w - 5):
            if lanels_slic[i, j] != lanels_slic[i, j + 1] and lanels_slic[i, j] == lanels_slic[i, j - 5] \
                or lanels_slic[i, j] != lanels_slic[i + 1, j] and lanels_slic[i, j] == lanels_slic[i - 5, j]:
                boundaries[i, j] = [255, 255, 0]  # 用红色标记边界
                boundaries[i, j + 1] = [255, 255, 0]  # 用红色标记边界
                # boundaries[i, j - 1] = [255, 255, 255]  # 用红色标记边界
            else:
                pass

    plt.imshow(boundaries)
    plt.axis('off')
    plt.show()
image  = Image.open('ndrishtiGS_085.png').convert('RGB')
image = color.rgb2lab(image)
image_np = np.array(image)
print(image_np.shape)

labels,centers = slic_method(image_np,1024,10,5)
image_np = color.lab2rgb(image_np)
image_np = (image_np * 255).astype(np.uint8)
show_slic_image(image_np,labels)

# show_slic_pooling(centers)


