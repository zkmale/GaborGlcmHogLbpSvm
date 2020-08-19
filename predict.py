# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:35:32 2020

@author: Administrator
"""
import glob
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
import cv2
import matplotlib.pyplot as plt
from hog_svm import pre_handle_imgs_gabor

test_feat_path = 'test/'
model_path = 'model/'
aa = glob.glob(os.path.join(test_feat_path, '*.feat'))
feat_path = aa[1]


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

gabor_kernel = [[7,20,5,3]]


def pre_handle_imgs(imgs):
    #imgs = cv2.imread(path)
    gray_img = cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)
    rotate_img = cv2.warpAffine(gray_img, matRotate, (rotate_x, rotate_y))
    fisheye_img = cv2.fisheye.undistortImage(rotate_img, K, D=D, Knew=Knew)
    cut_img = fisheye_img[340:470,5:700]
    dst = cv2.warpPerspective(cut_img, M, (695,130))
    dst = dst[0:110, 15:680]
    gaussian_img = cv2.GaussianBlur(dst,(7,7),11)
    resize_img = cv2.resize(gaussian_img, (140, 60))#注意这里的位置是先列后行 是point的格式
    return resize_img

def get_predict_feat(img_path, gabor_kernel):
    imgs = cv2.imread(img_path)
    image_arr = np.array(imgs)
    handle_img = pre_handle_imgs_gabor(image_arr, gabor_kernel)
    #gray = rgb2gray(image) / 255.0
    # print(gray)
    # 这句话根据你的尺寸改改
    #show(handle_img)
    fd = hog(handle_img, orientations=9,block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[6, 6], visualise=False,
             transform_sqrt=False)
    #print(len(fd))
    return fd


def get_one_result(img_path, model_path, gabor_kernel):
    clf = joblib.load(model_path+'day_medium_eral_innel_(7,20,5,3)_30')
    features = get_predict_feat(img_path, gabor_kernel)
    features = features.reshape((1, -1)).astype(np.float64)
    result = int(clf.predict(features)[0])
    return result
    



if __name__ == "__main__":
    path = r"g:\zk\hog_svm_extract_feature\mediu_test_imgs_eral_innel"
    #imgs_names = os.listdir(path)

    path = r"e:\zoomlion\data\gongqiWinding200706\day\adjustDatasets2\img00001.jpg" 
    result = get_one_result(path, model_path, gabor_kernel)
    print(result)
    
    """
    f_txt = open(r"g:\zk\hog_svm_extract_feature\mediu_test_imgs_eral_innel\medium_test_eral_innel.txt")
    imgname_labels = f_txt.readlines()
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    FPmt = []
    FNmat = []
    i = 1
    for imgs_name_label in imgname_labels:
        img_path = os.path.join(path, imgs_name_label[:-3])
        result = get_one_result(img_path, model_path, gabor_kernel)
        if result == int(imgs_name_label[-2]) == 1:
            TP += 1
        elif result == 1 and int(imgs_name_label[-2]) == 0:
            FP += 1
            FPmt.append(imgs_name_label[:-3])
        elif result == 0 and int(imgs_name_label[-2]) == 1:
            FN += 1
            FNmat.append(imgs_name_label[:-3])
        elif result == 0 and int(imgs_name_label[-2]) == 0:
            TN += 1
        if i % 100 == 0:
            recall = TP/(TP+FN+1e-7)
            precision = TP/(TP+FP+1e-7)
            acc = (TP+TN)/(TP+TN+FP+FN+1e-7)
            F1 = 2*precision*recall/(precision+recall+1e-7)
            print(F1)
            print(i)
            print("acc = ", acc)
        i += 1
        #print('{}'.format(imgs_name), result)
    """
    
    
    
    """
    clf = joblib.load(model_path+'model')
    path = "g:\\zk\\HOG_SVM-master\\imgs_cv"
    img_name_lists = os.listdir(path)
    result_list = []
    for img_name in img_name_lists:
        img_path = os.path.join(path, img_name)
        features = get_predict_feat(img_path)
        features = features.reshape((1, -1)).astype(np.float64)
        result = clf.predict(features)
        print(result)
        result_list.append(result)
    """



        
            