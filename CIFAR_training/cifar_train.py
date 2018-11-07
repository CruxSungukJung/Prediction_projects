import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.cifar10 import load_data


(x_train, y_train), (x_test, y_test) = load_data()
class Cnn_conv:
    filter_arr= [] 
    last_layer =None
    x_data = None
    initializer = None
    stride_arr =[]
    last_size = 1
    def make_filter_layers(self,num):
        pre_channel= 3
        
        for i in range(num):
            stride = int(input(str(i+1)+'th how many you want make stides?'))
            self.stride_arr.append([1,stride,stride,1])
            if i==0:
                filter_ = tf.Variable(self.initializer([4,4,pre_channel,4]))
                pre_channel = 4
                self.filter_arr.append(filter_)
            else:
                kernal_n = int(input(str(i+1)+'th how many you want make kernals?'))
                channel =  int(input(str(i+1)+'th how many you want make cnannels?'))
                filter_ = tf.Variable(self.initializer([kernal_n,kernal_n,pre_channel,channel]))
                self.filter_arr.append(filter_)
                pre_channel = channel
                
                
        
    def __init__(self,input_x,num_of_layers=2):
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.make_filter_layers(num_of_layers)
        
        for i in range(num_of_layers):
            if i==0:
                convol_unit = tf.nn.conv2d(input_x,filter=self.filter_arr[i],
                                           strides=self.stride_arr[i],padding='SAME')
            else:
                convol_unit = tf.nn.conv2d(self.last_layer,filter=self.filter_arr[i],
                                           strides=self.stride_arr[i],padding='SAME')
            convol_unit = tf.nn.relu(convol_unit)
            self.last_layer = convol_unit
        last_data_width = self.last_layer.shape[-3]
        last_data_height= last_data_width
        last_channel =self.last_layer.shape[-1]
        self.last_size= last_channel*last_data_height*last_data_width   
        
    def return_conv_output(self):
        return self.last_layer,int(self.last_size)
    

 
    
X = tf.placeholder(dtype=tf.float32,shape=[None,32*32*3])
Y = tf.placeholder(dtype=tf.float32,shape=[None,1])
