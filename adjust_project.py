# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:11:42 2020

@author: Administrator
"""
import cv2
import os
import numpy as np
from windingGQ_New import pre_handle_imgs, norm, count_Trough, show
from hog_svm import pre_handle_imgs_gabor



xuanze = False
if xuanze == False:
#    gabor = [[13, 4, 3, 3],[7,20,5,3]]
#    gabor = [[13, 4, 3, 3]]
#    gabor = [[7,20,5,3]]   #F1值达到90 held=30
#    gabor = [[13,22,6,2]]
#    gabor = [[13,12,7,4]]
    gabor = [[13,24,2,4]]
    path = r"e:\zoomlion\data\gongqiWinding200706\day\adjustDatasets2"
    imgs_names = os.listdir(path)
    for img_name in imgs_names:
        img_path = os.path.join(path, img_name)
        
        imgs = cv2.imread(img_path)
        ar = pre_handle_imgs_gabor(imgs, gabor)
        print(img_name)
        show(ar)
        
        """
        am2 = pre_handle_imgs(img_path)
        
        
#        ksize = 11
#        lambd = 14
#        sigma = 4
#        gamma = 1
#        theta = 0
#        psi = 1
        
        ksize = 9
        lambd = 5
        sigma = 2
        gamma = 2
        theta = 0
        psi = 1

        gabor_kernel = cv2.getGaborKernel((ksize,ksize), sigma, 0, 1/lambd, gamma, psi)
        img_gabor_real = cv2.filter2D(am2/255, -1, gabor_kernel)
        show(am2)
        show(img_gabor_real)
        line_pixel = norm(img_gabor_real[15])
        bogu1, index1 = count_Trough(line_pixel)
        
    #    print("min value:{},max value:{},mean value:{}".format(round(np.min(img_gabor_real),2),
    #                                                                     round(np.max(img_gabor_real),2),
    #                                                                    round(np.mean(img_gabor_real))))
        
        print(img_name)
        _,thresh_img = cv2.threshold(img_gabor_real,18,255,cv2.THRESH_BINARY_INV)
        #show(thresh_img)
        
        max_value = np.max(img_gabor_real)
        min_value = np.min(img_gabor_real)
        
        sum_img = np.uint8((img_gabor_real - min_value)*250.0/(max_value - min_value))
        
        mean_value = np.mean(sum_img)
    #    th = 220+(129 - mean_value)
    #    print(th)
    #    _,thresh_img1 = cv2.threshold(sum_img, 225 ,255,cv2.THRESH_BINARY_INV)
        _,thresh_img1 = cv2.threshold(sum_img, 220 ,255,cv2.THRESH_BINARY)
        show(thresh_img1)
        erode_img = cv2.erode(thresh_img1,np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8),iterations=1)
        show(erode_img)
        
        print("min value:{},max value:{},mean value:{}".format(round(np.min(sum_img),2),
                                                                         round(np.max(sum_img),2),
                                                                         round(np.mean(sum_img))))
       
        """

      
else:
#    path = r"e:\zoomlion\data\gongqiWinding200706\day\adjustDatasets1\img00232.jpg"   
#    path = r"e:\zoomlion\data\gongqiWinding200706\day\adjustDatasets2\img00067.jpg" 
    path = r"e:\zoomlion\data\gongqiWinding200706\day\adjustDatasets2\img00236.jpg"
    #am2 = pre_handle_imgs(path)
    imgs = cv2.imread(path)
    #gabor = [[7,20,5,3],[11, 21, 4, 2]]
    #gabor = [[11, 21, 4, 2]]
    gabor = [[7,20,5,3]]
    ar = pre_handle_imgs_gabor(imgs, gabor)
    show(ar)

    
    
    #for si in range(0,10,1):
    for sigma in range(2,25,1):
        for lambd in range(3,25,1):
            for gamma in [1,2,3,4]:
                for ksize in [7,9,11,13]:
                    print(ksize,lambd,sigma,gamma,1,0)
#                    print(ksize,4+si*0.2,lambd,gamma,1,0)
                    
                    gabor = [[ksize, lambd, sigma, gamma]]
                    ar = pre_handle_imgs_gabor(imgs, gabor)
                    show(ar)
                    
                    """
                    gabor_kernel = cv2.getGaborKernel((ksize,ksize), sigma, 0, lambd, gamma, 1)
                    img_gabor_real = cv2.filter2D(am2/255, -1, gabor_kernel)
                    #show(am2)
                    
                    max_value = np.max(img_gabor_real)
                    min_value = np.min(img_gabor_real)
                    sum_img = np.uint8((img_gabor_real - min_value)*250.0/(max_value - min_value))
                    _,thresh_img1 = cv2.threshold(sum_img, 220 ,255,cv2.THRESH_BINARY)
                    erode_img = cv2.erode(thresh_img1,np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8),iterations=1)
                    
                    show(erode_img)
                    show(img_gabor_real)
                    cv2.waitKey(90)
                    #line_pixel = norm(img_gabor_real[15])
                    #bogu1, index1 = count_Trough(line_pixel)
                    """
                    
                    
    """
    #img990
#    ksize = 7
#    lambd = 20
#    sigma = 5
#    gamma = 3
#    theta = 0
#    psi = 1
    
#img239
    ksize = 13
    lambd = 4
    sigma = 3
    gamma = 3
    theta = 0
    psi = 1
    
    gabor_kernel = cv2.getGaborKernel((ksize,ksize), sigma, 0, lambd, gamma, 1)
    img_gabor_real = cv2.filter2D(am2/255, -1, gabor_kernel)
    show(am2)
    show(img_gabor_real)
    
#    line_pixel = norm(img_gabor_real[15])
#    bogu1, index1 = count_Trough(line_pixel)
    
#    print("min value:{},max value:{},mean value:{}".format(round(np.min(img_gabor_real),2),
#                                                                     round(np.max(img_gabor_real),2),
#                                                                    round(np.mean(img_gabor_real))))
    
    print(img_name)
    _,thresh_img = cv2.threshold(img_gabor_real,18,255,cv2.THRESH_BINARY_INV)
    #show(thresh_img)
    
    max_value = np.max(img_gabor_real)
    min_value = np.min(img_gabor_real)
    
    sum_img = np.uint8((img_gabor_real - min_value)*250.0/(max_value - min_value))
    
    mean_value = np.mean(sum_img)
#    th = 220+(129 - mean_value)
#    print(th)
    _,thresh_img1 = cv2.threshold(sum_img, 220 ,255,cv2.THRESH_BINARY)
    erode_img = cv2.erode(thresh_img1,np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8),iterations=1)
    show(erode_img)
    
    
    print("min value:{},max value:{},mean value:{}".format(round(np.min(sum_img),2),
                                                                     round(np.max(sum_img),2),
                                                                     round(np.mean(sum_img))))
 """


