import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from walk_or_run_get_test_train import *
import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression



### image data shape is (224,224,4)
with tf.Session() as sess:
    
    X,Y= sess.run(train_img) ,train_labels
    X = np.reshape(X,[-1,224,224,4])
    Y = np.array(Y)
    X_test, Y_test= sess.run([test_img]),test_label
    X_test=np.reshape(X_test,[-1,224,224,4])
    Y_test= np.array(Y_test)
    
    



def CNN_():
    CNN = input_data(shape=[None, 224, 224, 4], name='input_x')
    CNN = conv_2d(CNN,32,4,activation='relu', regularizer="L2")
    CNN = max_pool_2d(CNN,2)
    CNN = dropout(CNN,keep_prob=0.5)
    CNN = local_response_normalization(CNN)
    
    #layer 1 size:56
    CNN = conv_2d(CNN,5,3,activation='relu', regularizer="L2")
    CNN = max_pool_2d(CNN,2)
    CNN = dropout(CNN,keep_prob=0.5)
    CNN = local_response_normalization(CNN)
    ###layer 2 size 56
        #layer 1 size:56    
    
    CNN = conv_2d(CNN,5,2,activation='relu', regularizer="L2")
    CNN = max_pool_2d(CNN,2)
    CNN = dropout(CNN,keep_prob=0.5)
    #layer 3 size :28
    return CNN

fc_layer = fully_connected(CNN_(),2,activation='softmax')
output = regression(fc_layer,optimizer='adam',learning_rate=0.00000089,loss='categorical_crossentropy',name='targets')

model = tflearn.DNN(output,tensorboard_verbose=0,tensorboard_dir = './walk_run',checkpoint_path = './walk_run/checkpoint')
model.fit({'input_x':X},{'targets':Y},show_metric=True,n_epoch=50,validation_set=({'input_x':X_test},{'targets':Y_test}),batch_size=600)
model.evaluate({'input_x':X_test},{'targets':Y_test})