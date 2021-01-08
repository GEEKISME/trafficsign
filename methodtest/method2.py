# 导入需要的包
import sys
from matplotlib import pyplot
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.utils import plot_model
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model.traffic_network import Lenet

tfversion = tf.__version__
print(tfversion)
kerasversion = keras.__version__
print(kerasversion)
print("=============================")
channel = 3
height = 32
width = 32
class_num = 62
norm_size = 32#参数
batch_size = 32
# epochs = 40
epochs = 20

model = Lenet.neural(channel=channel, height=height,
                         width=width, classes=class_num)#网络
model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",metrics=["accuracy"])#配置

plot_model(model,to_file='traffic_model.png',show_shapes=True,show_layer_names=True)