# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:11:45 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:17:44 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:12:00 2019

@author: Administrator
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mping
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import cv2
import os
from skimage import color
from scipy.signal import convolve2d
import imageio
from skimage import exposure
from PIL import Image,ImageEnhance
from predict import get_one_result


#鱼眼畸变矫正参数初始化
K = np.array([[640,   0, 420],
              [   0, 600, 400],
              [   0,   0,   1]],
              dtype=np.float32)
Knew = K.copy()
D = np.zeros(4, dtype=np.float32)
Knew[(0,1), (0,1)] = 0.86 * Knew[(0,1), (0,1)]

#自适应直方均衡初始化
clache_1 = cv2.createCLAHE(clipLimit=5,tileGridSize=(12,12))
clache_2 = cv2.createCLAHE(clipLimit=3,tileGridSize=(6,6))

#仿射变换矩阵初始化
pts1 = np.float32([[0 , 0], [705, 0], [0 , 293], [705, 293]])
pts2 = np.float32([[0 , 0], [705, 0], [60, 293], [665, 293]])
M = cv2.getPerspectiveTransform(pts1, pts2)

#图像旋转初始化
img_x, img_y = 720, 480
center_x, center_y = img_x/2, img_y/2
rotate_x, rotate_y = img_x, img_y
angle, scale = 1.8, 1
matRotate = cv2.getRotationMatrix2D((center_x, center_y),
                                    angle,
                                    scale)#注意这里的位置是先行后列 是python的格式

def show(image):
    plt.imshow(image,cmap='gray')
    # plt.xticks([]),plt.yticks([])
    plt.show()


def norm(x):
    #该函数是对数据进行归一化
    max_x = max(x)
    min_x = min(x)
    return [int(round((i - min_x)/(max_x - min_x)*200)) for i in x]

"""
def count_Trough(x):
    #该函数主要计算波谷的个数、波谷的方差、波谷值和波谷下标
    num_x = len(x)
#    ave_x = sum(x)//num_x
    bogu = []
    index = []
    count = 0
    threshold = sorted(x)[62]
    #设置一个阈值，波谷的最大值，如果超过阈值则不能被看作波谷
    for i in range(2,num_x - 2):
        if x[i] <= x[i - 1] and x[i] <= x[i - 2]and x[i] <= x[i + 1] and x[i] <= x[i + 2] and x[i] < threshold:
            #判断是否为波谷
            count += 1
            bogu.append(x[i])
            index.append(i)
    bogu_var = np.var(bogu)
    return count, bogu_var, bogu, index
"""
def cout_var(lists):
    lists_sorted = sorted(lists)
    lists_var = np.var(lists_sorted[3:-3])
    return lists_var
def cout_vague(x):
    x1 = x[5:35]
    x2 = x[45:75]
    x3 = x[85:115]
    x4 = x[125:155]
    x1_var = cout_var(x1)
    x2_var = cout_var(x2)
    x3_var = cout_var(x3)
    x4_var = cout_var(x4)
    #print("x1_var = ", x1_var)
    #print("x2_var = ", x2_var)
    #print("x3_var = ", x3_var)
    #print("x4_var = ", x4_var)
    vague = int(min(x1_var,x2_var,x3_var,x4_var) < 15)
    return vague

def isInnerRing(x):
     x_norm = norm(x)
     x_norm = np.array(x_norm)
     if sum(x_norm > 90) > 5:
         return 1
     return 0
        


def count_Trough(x):
    #该函数主要计算波谷的个数、波谷的方差、波谷值和波谷下标
    num_x = len(x)
#    ave_x = sum(x)//num_x
    bogu = []
    index = []
    """
    sorted_x = sorted(x)
    var_list = sorted_x[25:30]+sorted_x[-30:-25]
    var_x = np.var(var_list)
    """
    threshold = sorted(x)[62]
    #设置一个阈值，波谷的最大值，如果超过阈值则不能被看作波谷
    for i in range(2,num_x - 2):
        if x[i] <= x[i - 1] and x[i] <= x[i - 2] and x[i] <= x[i + 1] and x[i] <= x[i + 2] and x[i] < threshold:
#        if x[i] <= x[i - 1] and x[i - 1] <= x[i - 2] and x[i] <= x[i + 1] and x[i + 1] <= x[i + 2] and x[i] < threshold:
            #判断是否为波谷
            bogu.append(x[i])
            index.append(i)
    return bogu, index

"""
def pre_handle_imgs(path):
    imgs = cv2.imread(path)
    gray_img = cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)
    show(gray_img)
    rotate_img = cv2.warpAffine(gray_img, matRotate, (rotate_x, rotate_y))
    fisheye_img = cv2.fisheye.undistortImage(rotate_img, K, D=D, Knew=Knew)
    show(fisheye_img)
    cut_img = fisheye_img[340:470,5:700]
    dst = cv2.warpPerspective(cut_img, M, (695,130))
    dst = dst[0:110, 15:680]
    gaussian_img = cv2.GaussianBlur(dst,(7,7),11)
    resize_img = cv2.resize(gaussian_img, (140, 60))#注意这里的位置是先列后行 是point的格式
    
    adap_img = clache_1.apply(resize_img)
    adap_img = clache_2.apply(adap_img)
    std_img = cv2.medianBlur(np.uint8(adap_img),3)
    
    return std_img
"""    

def pre_handle_imgs(path):
    imgs = cv2.imread(path)
    gray_img = cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)
    #show(gray_img)
    rotate_img = cv2.warpAffine(gray_img, matRotate, (rotate_x, rotate_y))
    fisheye_img = cv2.fisheye.undistortImage(rotate_img, K, D=D, Knew=Knew)
    #show(fisheye_img)
    cut_img = fisheye_img[340:470,5:715]
    dst = cv2.warpPerspective(cut_img, M, (710,130))
    #dst = dst[0:110, 15:680]
    gaussian_img = cv2.GaussianBlur(dst,(7,7),11)
    resize_img = cv2.resize(gaussian_img, (140, 60))#注意这里的位置是先列后行 是point的格式
    adap_img = clache_1.apply(resize_img)
    adap_img = clache_2.apply(adap_img)
    #std_img = cv2.medianBlur(np.uint8(adap_img),3)
    recut_img = adap_img[:,6:131]
    return recut_img

#x[i] - x[i - 1] <= 3 or 6<=
def distance_Trough(x):
    #计算波谷之间的像素点不在正常范围内的个数
    dist = 0
    lens = len(x)
#    kq = []
    for i in range(1,lens):
#        if 8 <= x[i] - x[i - 1] <= 10 or x[i] - x[i - 1] <= 2 or 16 <= x[i] - x[i - 1] <= 18:
        if x[i] - x[i - 1] <= 8 or 13 <= x[i] - x[i - 1] <= 15:
            continue
        dist += 1
#        kq.append(x[i] - x[i - 1])
#    print(kq)
    return dist


def Rcognition_roller(line_pixel):
    #通过裁剪图片后每一行像素来识别滚轮绳是否打结
    line_pixel = norm(line_pixel)
    bogu1, index1 = count_Trough(line_pixel)
    #print("index ", index1)
    dists = distance_Trough(index1)
#    print(dists,kq)
#    if dists <= 2:
#        return 1
    
    if dists == 2:
        return 0.3
    elif dists == 3:
        return 1 
    elif dists == 4:
        return 1.3 
    elif dists >= 5:
        return 1.6 
    
#    if dists == 0:
#        return 1      
    return 0 


# this function is for read image,the input is directory name
def read_directory(directory_name):
    array_of_img = [] # this if for store all of the image data
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        #print(img)
        #print(array_of_img)

def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    return dst


"""
def my_algorithm(imgs_path):
    #如果乱绳显示为1，正常显示为0
    imgs = mping.imread(imgs_path)
    imgs_crop = imgs[285:545,465:742]
    imgs_gray=cv2.cvtColor(imgs_crop, cv2.COLOR_BGR2GRAY)
    nums = len(imgs_gray)
    pos = 0
    reg = 0
    for i in range(nums):
        if Rcognition_roller(imgs_gray[i]) == 1:
            pos += 1
            continue
        reg += 1
    if reg <= 89:
        print(reg)
        return 0
    return 1
"""


"""
#不做图像预处理
def my_algorithm_pos(imgs_path):
    #如果乱绳显示为1，正常显示为0
    imgs = mping.imread(imgs_path)
    imgs_crop = imgs[285:545,465:742]
    imgs_gray=cv2.cvtColor(imgs_crop, cv2.COLOR_BGR2GRAY)
#    imgs_gray = imgs_crop
    nums = len(imgs_gray)
    reg = 0
    for i in range(nums):
        reg += Rcognition_roller(imgs_gray[i])
#    return reg
    
#    if reg >= 20:
#        kk.append(imgs_path.split("\\")[-1].split('.')[0]) 
    print(reg)
    if reg <= 22:       
        return 0
    return 1
""" 


#不做图像预处理
def my_algorithm_pos_1(imgs_path, model_path, continue_normal = 12, reg_score = 2.2):
    #如果乱绳显示为1，正常显示为0
    #imgs = mping.imread(imgs_path)
    imgs = pre_handle_imgs(imgs_path)
    
#    imgs_gray = imgs_crop
    nums = len(imgs)
    n = 0
#    max_n = []
    reg = 0
    for i in range(nums//4):
        score = Rcognition_roller(imgs[i*4])
        reg += score
        #print(score)
        if score == 0:
            n += 1
            if n >= continue_normal:
                return 0
        else:
#            max_n.append(n)
            n = 0
            continue
#    return reg
#    print("***", max(max_n))
    #print(111)
    #print("****",reg)
    if reg <= reg_score:
#        print(reg)
        return 0
    result = get_one_result(imgs_path, model_path)
        
#    print(reg)
    return result


"""
f = open("e:\\zoomlion\\data\\gongqiWinding200706\\day\\bigDatasets\\test\\y_Test.txt")
test_data = f.readlines()
f.close()
TP = 0
FP = 0
FN = 0
TN = 0
other = 0
othermat = []
FPmt = []
FNmat = []
model_path = 'model/'
count = 0
paths = "e:\\zoomlion\\data\\gongqiWinding200706\\day\\bigDatasets\\test\\imgs"
paths_name_lists = os.listdir(paths)
for i in range(len(test_data)):
    path = "e:\\zoomlion\\data\\gongqiWinding200706\\day\\bigDatasets\\test\\imgs\\img{}.jpg".format((5-len(str(i+1)))*'0'+str(i+1))

#   a,b = my_algorithm_pos_1(path)
#   count = count + b
    
    if my_algorithm_pos_1(path,model_path) == int(test_data[i]) == 1:
        TP += 1
    elif my_algorithm_pos_1(path,model_path) == 1 and int(test_data[i]) == 0:
        FP += 1
        FPmt.append(paths_name_lists[i])
    elif my_algorithm_pos_1(path,model_path) == 0 and int(test_data[i]) == 1:
        FN += 1
        FNmat.append(paths_name_lists[i])
    elif my_algorithm_pos_1(path,model_path) == 0 and int(test_data[i]) == 0:
        TN += 1
    else:
        other += 1
        othermat.append(i)
    if i % 100 == 0:       
        recall = TP/(TP+FN+1e-7)
        precision = TP/(TP+FP+1e-7)
        acc = (TP+TN)/(TP+TN+FP+FN+1e-7)
        F1 = 2*precision*recall/(precision+recall+1e-7)
        print(F1)
        print(i)
"""


#score = Rcognition_roller(imgs_gray[0])
#
#bogu1, index1 = count_Trough(line_pixels)
#dists = distance_Trough(index1)

#re = my_algorithm_pos_1(path)




"""
num = len(am)
mats2 = []
for i in range(num):
    mats2.append(am[i])
mat = np.array(mats2)
"""





