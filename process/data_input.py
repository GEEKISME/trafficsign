#!/usr/bin/env python
# encoding: utf-8
from keras.preprocessing.image import img_to_array#图片转为array
from keras.utils import to_categorical#相当于one-hot
from imutils import paths
import cv2
import numpy as np
import random
import os
#  path ： ‘../data/train’ ，norm_size 是32， class_num 是 62
def load_data(path,norm_size,class_num):
    data = []#数据x
    label = []#标签y
    # files 是一个 generator，这个generator是以path下的各个文件夹里的图片文件为源
    files = paths.list_images(path)
    # filelist是train下面是所有文件夹里的图片文件的路径集合,是一个大小为4572的list，每个元素类似于这样：../data/train\\00000\\01153_00000.jpg
    filelist = list(files)
    # sorted 操作其实没干啥image_paths与filelist 的元素一样 是 train下的所有图像文件的路径，共有4572张图片，每个路径的格式：‘../data/train/00000/01153_00000.png’
    image_paths = sorted(filelist)#imutils模块中paths可以读取所有文件路径
    random.seed(0)#保证每次数据顺序一致
    # 下面这行代码运行过后image_paths的里面的所有图片路径将打乱
    random.shuffle(image_paths)#将所有的文件路径打乱
    #
    for each_path in image_paths:    #each_path 是‘../data/train\\00038\\00653_00002.png’
        # image 是 ndarray (57,54,3) 的三维张量
        image = cv2.imread(each_path)#读取文件
        image = cv2.resize(image,(norm_size,norm_size))#统一图片尺寸 #image 现在是（32,32,3）的张量
        image = img_to_array(image)  #image是ndarray
        data.append(image) #data 是个list,其中的每个元素是（32，32,3）的tensor
        maker = int(each_path.split(os.path.sep)[-2])#切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        label.append(maker)    #label 是一个list
    # 图像是彩色的rgb,data数组是像素数*3，数组中每个的值在0-255之间，对数组中的每个值除以255来进行归一化
    # 归一化  data是一个list，长度4572，每个元素是(32,32,3)的tensor,首先将data由list变为ndarray,然后进行归一化
    data = np.array(data,dtype="float")/255.0  # data 是 （4572,32,32,3）的ndarray
    label = np.array(label)  #label也有由list变为ndarray
    # to_categorical 是从 keras中导入的库
    label = to_categorical(label,num_classes=class_num)#one-hot   label变为4572行，62列的ndarray,每一行都是one-hot  label 是 （4572,62）的ndarray
    return data,label
