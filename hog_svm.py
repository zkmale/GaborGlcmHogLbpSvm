# -*- coding=utf-8 -*-
import glob
import platform
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC,NuSVC,SVC
import shutil
import sys
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from skimage import feature as skif

sys.path.append("..")
from GLCM import glcm
from durfert.gqGetSvmModel import windingSystemTrainTool_class as train_tool 






# 第一个是你的类别   第二个是类别对应的名称   输出结果的时候方便查看
label_map = {0: 'normal',
             1: 'innormal'
             }
# 训练集图片的位置
train_image_path = 'mediu_train_imgs_eral_innel'
# 测试集图片的位置
test_image_path = 'mediu_test_imgs_eral_innel'

# 训练集标签的位置
train_label_path = os.path.join('mediu_train_imgs_eral_innel','medium_train_big_test_eral_innel.txt')
# 测试集标签的位置
test_label_path = os.path.join('mediu_test_imgs_eral_innel','medium_test_eral_innel.txt')

image_height = 128
image_width = 100

train_feat_path = 'train_medium_day_train/'
test_feat_path = 'test_medium_day_test/'
model_path = 'model/'

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


def train_data_shuffle(train_data, train_label):
    # =============================================================================
    # train data shuffle
    # =============================================================================
    #assert train_data.shape[0] == train_label.shape[0]
    
    #train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[1]))
    train_image_num = len(train_data)
    train_image_index = np.arange(train_image_num)
    np.random.shuffle(train_image_index)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_data = train_data[train_image_index]
    train_label = train_label[train_image_index]

    return train_data,train_label


def get_hogpy_feature(img, label):
    fd = hog(img, orientations=11,block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[3, 3], visualise=False,
                 transform_sqrt=False)
    fd = np.concatenate((fd, [label]))
    return fd

def get_hogcv_feature(img, label):
    winsize = 12
    winSize = (winsize,winsize)
    blockSize = (int(winsize/2),int(winsize/2))
    blockStride = (int(winsize/4),int(winsize/4))
    cellSize = (int(winsize/4),int(winsize/4))
    #cellSize = (3,3)
    nbins = 10
    derivAperture = 3
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.5
    gammaCorrection = 0
    nlevels = 72
    hog = cv2.HOGDescriptor(winSize,
                            blockSize,
                            blockStride,
                            cellSize,
                            nbins,
                            derivAperture,
                            winSigma,
                            histogramNormType,
                            L2HysThreshold,
                            gammaCorrection,
                            nlevels)
    
    winStride = (int(winsize/4),int(winsize/4))
    padding = (int(winsize/4),int(winsize/4))

    locations = train_tool.windingSystemTrainTool_class().getHogLocations(img.shape[1], img.shape[0], winsize)
    features = hog.compute(img,winStride,padding,locations).flatten()
    fd = np.concatenate((features, [label]))
    
    return fd

def get_lbp_feature(img, label):
    lbp = skif.local_binary_pattern(img, 8, 1, 'default')
    max_bins = int(lbp.max() + 1)             
    hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))     
    fd = np.concatenate((hist, [label]))
    return fd

def pre_handle_imgs(imgs):
    #imgs = cv2.imread(path)
    gray_img = cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)
    rotate_img = cv2.warpAffine(gray_img, matRotate, (rotate_x, rotate_y))
    fisheye_img = cv2.fisheye.undistortImage(rotate_img, K, D=D, Knew=Knew)
#    cut_img = fisheye_img[340:470,5:715]
    cut_img = fisheye_img[340:470,:]
    dst = cv2.warpPerspective(cut_img, M, (720,130))
    #dst = dst[0:110, 15:680]
    gaussian_img = cv2.GaussianBlur(dst,(7,7),11)
    resize_img = cv2.resize(gaussian_img, (140, 60))#注意这里的位置是先列后行 是point的格式
    adap_img = clache_1.apply(resize_img)
    adap_img = clache_2.apply(adap_img)
    #std_img = cv2.medianBlur(np.uint8(adap_img),3)
    recut_img = adap_img[:,6:131]
    
    return recut_img


def pre_handle_imgs_gabor(imgs, gabor_kernel):
    #imgs = cv2.imread(path)
    whole_sum_img = np.zeros((60,125))
    
    recut_img = pre_handle_imgs(imgs)
    for [ksize, lambd, sigma, gamma] in gabor_kernel:
        #show(recut_img)
        gabor_kernel = cv2.getGaborKernel((ksize,ksize), sigma, 0, lambd, gamma, 1)
        img_gabor_real = cv2.filter2D(recut_img/255, -1, gabor_kernel)
        whole_sum_img += img_gabor_real
    max_value = np.max(whole_sum_img)
    min_value = np.min(whole_sum_img)
    sum_img = np.uint8((whole_sum_img - min_value)*250.0/(max_value - min_value))
    #_,thresh_img1 = cv2.threshold(sum_img,75,255,cv2.THRESH_BINARY)
    #erode_img = cv2.erode(thresh_img1,np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8),iterations=1)
    
    return sum_img


def pre_handle_imgs_glcm(imgs):

    """
    gray_img = cv2.cvtColor(imgs,cv2.COLOR_BGR2GRAY)
    rotate_img = cv2.warpAffine(gray_img, matRotate, (rotate_x, rotate_y))
    fisheye_img = cv2.fisheye.undistortImage(rotate_img, K, D=D, Knew=Knew)
    cut_img = fisheye_img[340:470,5:715]
    dst = cv2.warpPerspective(cut_img, M, (710,130))
    #dst = dst[0:110, 15:680]
    gaussian_img = cv2.GaussianBlur(dst,(7,7),11)
    resize_img = cv2.resize(gaussian_img, (140, 60))#注意这里的位置是先列后行 是point的格式
    adap_img = clache_1.apply(resize_img)
    adap_img = clache_2.apply(adap_img)
    #std_img = cv2.medianBlur(np.uint8(adap_img),3)
    recut_img = adap_img[:,6:131]
    """
    recut_img = pre_handle_imgs(imgs)
    h, w = np.shape(recut_img)

#    mean = glcm.fast_glcm_mean(recut_img, ks=9)
#    std = glcm.fast_glcm_std(recut_img, ks=9)
#    cont = glcm.fast_glcm_contrast(recut_img, ks=9)
#    diss = glcm.fast_glcm_dissimilarity(recut_img, ks=9)
#    homo = glcm.fast_glcm_homogeneity(recut_img, ks=9)
#    asm, ene = glcm.fast_glcm_ASM(recut_img, ks=9)  #0.879
#    ma = glcm.fast_glcm_max(recut_img, ks=9)        #0.883
    ent = glcm.fast_glcm_entropy(recut_img, vmax=250, nbit=8, ks=9)    #0.883
    #show(recut_img)
    
    max_value = np.max(ent)
    min_value = np.min(ent)
    sum_img = np.uint8((ent - min_value)*250.0/(max_value - min_value))
    #_,thresh_img1 = cv2.threshold(sum_img,75,255,cv2.THRESH_BINARY)
    #erode_img = cv2.erode(thresh_img1,np.array([[0,1,0],[0,1,0],[0,1,0]],np.uint8),iterations=1)
    
    return sum_img




# 获得图片列表
def get_image_list(filePath, nameList):
    print('read image from ',filePath)
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath,name))
        
        img_list.append(temp.copy())
        temp.close()
    return img_list


# 提取特征并保存
def get_feat(image_list, name_list, label_list, savePath, isGabor_Glcm = "glcm", isHog_Lbp = "hog"):
    i = 0
    #gabor_kernel = [[13,24,2,4]]
    gabor_kernel = [[7,20,5,3]]
    for image in image_list:
        """
        try:
            # 如果是灰度图片  把3改为-1
            image = np.reshape(image, (image_height, image_width, 3))
        except:
            print('发送了异常，图片大小size不满足要求：',name_list[i])
            continue
        """
        image_arr = np.array(image)
        if isGabor_Glcm == "gabor":
            handle_img = pre_handle_imgs_gabor(image_arr,gabor_kernel)
        elif isGabor_Glcm == "glcm":
            handle_img = pre_handle_imgs_glcm(image_arr)
        else:
            print("不好意思，您的选择错误，请选择gabor或者glcm")
        #gray = rgb2gray(image) / 255.0
        # print(gray)
        # 这句话根据你的尺寸改改
        #show(handle_img)
        
        if isHog_Lbp == "hog":
            #提取hog特征
            fd = get_hogcv_feature(handle_img, label_list[i])
        elif isHog_Lbp == "lbp":
            #提取lbp特征
            lbp = skif.local_binary_pattern(handle_img, 8, 1, 'default')
            max_bins = int(lbp.max() + 1)             
            hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))     
            fd = np.concatenate((hist, [label_list[i]]))
        else:
            print("不好意思，您的选择错误，请选择hog或者lbp")

        
        
        fd_name = name_list[i] + '.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path)
        i += 1
    print("Test features are extracted and saved.")


# 变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


# 获得图片名称与对应的类别
def get_name_label(file_path):
    print("read label from ",file_path)
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            #一般是name label  三部分，所以至少长度为3  所以可以通过这个忽略空白行
            if len(line)>=3: 
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].replace('\n','').replace('\r',''))
                if not str(label_list[-1]).isdigit():
                    print("label必须为数字，得到的是：",label_list[-1],"程序终止，请检查文件")
                    exit(1)
    return name_list, label_list


# 提取特征
def extra_feat():
    train_name, train_label = get_name_label(train_label_path)
    train_name, train_label = train_data_shuffle(train_name, train_label)
    
    test_name, test_label = get_name_label(test_label_path)
    

    train_image = get_image_list(train_image_path, train_name)#获取所有图片的绝对路径列表
    test_image = get_image_list(test_image_path, test_name)#获取测试图片的绝对路径列表
    get_feat(train_image, train_name, train_label, train_feat_path)
    get_feat(test_image, test_name, test_label, test_feat_path)


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)


# 训练和测试
def train_and_test():
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])
    print("Training a Linear LinearSVM Classifier.")
    
    """
    features = np.array(features,np.float64)
    pca1 = PCA(n_components=0.9)
    pca1.fit(features)
    features = pca1.transform(features)
    #print(features.shape)
    """
    
    clf = LinearSVC()
    clf.fit(features, labels)
    

    # 下面的代码是保存模型的
    if not os.path.exists(model_path):
        os.makedirs(model_path)   
    joblib.dump(clf, model_path + 'model_day_medium_train_eral_innel')
    # 下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
    # clf = joblib.load(model_path+'model')
    print("训练之后的模型存放在model文件夹中")
    # exit()
    result_list = []
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        
        """
        data_test_feat = np.array(data_test_feat,np.float64)
        data_test_feat = pca1.transform(data_test_feat)
#        data_test_feat = data_test_feat.reshape((-1,1))
        """
        
        result = clf.predict(data_test_feat)
        result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
        if int(result[0]) == int(data_test[-1]):
            correct_number += 1
    write_to_txt(result_list)
    rate = float(correct_number) / total
    t1 = time.time()
    print('准确率是： %f' % rate)
    print('耗时是 : %f' % (t1 - t0))


def write_to_txt(list):
    with open('result.txt', 'w') as f:
        f.writelines(list)
    print('每张图片的识别结果存放在result.txt里面')


if __name__ == '__main__':

    mkdir()  # 不存在文件夹就创建
    # need_input = input('是否手动输入各个信息？y/n\n')

    # if need_input == 'y':
    #     train_image_path = input('请输入训练图片文件夹的位置,如 /home/icelee/image\n')
    #     test_image_path = input('请输入测试图片文件夹的位置,如 /home/icelee/image\n')
    #     train_label_path = input('请输入训练集合标签的位置,如 /home/icelee/train.txt\n')
    #     test_label_path = input('请输入测试集合标签的位置,如 /home/icelee/test.txt\n')
    #     size = int(input('请输入您图片的大小：如64x64，则输入64\n'))
#    if sys.version_info < (3,):
#        need_extra_feat = raw_input('是否需要重新获取特征？y/n\n')
#    else:
#        need_extra_feat = input('是否需要重新获取特征？y/n\n')
#
#    if need_extra_feat == 'y':
    shutil.rmtree(train_feat_path)
    shutil.rmtree(test_feat_path)
    mkdir()
    extra_feat()  # 获取特征并保存在文件夹

    train_and_test()  # 训练并预测





