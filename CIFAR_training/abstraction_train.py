import tensorflow as  tf
import numpy as np
from tensorflow.contrib import learn
from sklearn.datasets import load_boston
from sklearn import datasets,metrics,preprocessing
boston = load_boston()

x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target
x = tf.placeholder(tf.float64,shape = [None,13])
y_true = tf.placeholder(tf.float64,shape = None)

with tf.name_scope('inference') as scope:
    w = tf.Variable(tf.zeros([1,13],dtype=tf.float64,name="weight"))
    b = tf.Variable(0,dtype=tf.float64,name="bias")
    y_pred = tf.matmul(w,tf.transpose(x))+b
    
with tf.name_scope('loss') as scope:
    cost = tf.reduce_mean(tf.square(y_true-y_pred))
    
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(optimizer, feed_dict = {x:x_data,y_true:y_data})
        mse = sess.run(cost,{x:x_data,y_true:y_data})
        
        print(mse)
        