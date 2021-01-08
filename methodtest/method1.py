import random
import os
from imutils import paths
import numpy as np
from keras.utils import to_categorical#相当于one-hot

data = []#数据x
label = []#标签y
image_paths = sorted(list(paths.list_images("../data/test")))#imutils模块中paths可以读取所有文件路径
print(len(image_paths)) #image_paths 的长度是 2520 ，也就是test文件加下的62 的交通标志文件夹里共有2520张图片
print(image_paths)# image_paths 的元素的样子是类似这样的： ../data/test\\00000\\00017_00000.png
print("===========================================================================================")
random.seed(0)#保证每次数据顺序一致
random.shuffle(image_paths)#将所有的文件路径打乱
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(image_paths)# 这是打乱之后的imagepath，
for each_path in image_paths:
    maker = int(each_path.split(os.path.sep)[-2])  # 切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
    print(maker)
    label.append(maker)
print("=======================================")
print(label)# label里存储的就是每张图片的label,label 里的每个的值显然在0-61 之间
print("****************************************")
label = np.array(label)
print(label)# 将label变为数组形式
label = to_categorical(label,num_classes=62)#one-hot
print("-----------------------------------------")#label变为了2520行，62列的矩阵。 假设label中有一行的的第18个位置是1，那么代表与之对应的图片属于18这个分类
print(label)