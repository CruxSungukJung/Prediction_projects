import os
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import cv2
import glob
from keras import layers
from keras import models

PATH=""
train_run = os.path.join(PATH,'train','run')
train_run = glob.glob(os.path.join(train_run,"*.png"))

train_walk = os.path.join(PATH,'train','walk')
train_walk = glob.glob(os.path.join(train_walk,"*.png"))

train = pd.DataFrame()
train['file'] = train_run + train_walk
train.head()
train = train['file'].values.tolist()

##train and train labels

train_img = [cv2.imread(data)for data in train]
train_img = np.asarray(train_img,dtype=np.int64)

train_label = [1]*len(train_run)+[0]*len(train_walk)
train_label = np.reshape(train_label,[-1,1])

val_img = train_img[:200]
val_labels=train_label[:200]
train_data = train_img[200:]
train_labels = train_label[200:]
test_run = os.path.join(PATH,'test','run')
test_run = glob.glob(os.path.join(test_run,'*.png'))
test_walk = os.path.join(PATH,'test','walk')
test_walk = glob.glob(os.path.join(test_walk,'*.png'))

test_label =[1]*len(test_run) + [0]*len(test_walk)
test_label = np.reshape(test_label,[-1,1])
test = pd.DataFrame()
test['label'] = test_run + test_walk
test.head()
test =test['label'].values.tolist()

test_img = [cv2.imread(data) for data in test]
test_img = np.asarray(test_img,dtype=np.int64)

##test and test labels

model = models.Sequential()
model.add(layers.Conv2D(10,kernel_size=(1,1),padding='same',activation='relu',activity_regularizer='l2',input_shape=(224,224,3,)))
model.add(layers.Conv2D(64,kernel_size=(5,5),padding='same',activation='relu',activity_regularizer='l2'))
model.add(layers.MaxPool2D(pool_size=(4,4),padding='valid'))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(10,kernel_size=(1,1),padding='same',activation='relu',activity_regularizer='l2',input_shape=(224,224,3,)))
model.add(layers.Conv2D(64,kernel_size=(5,5),padding='same',activation='relu',activity_regularizer='l2'))
model.add(layers.Conv2D(32,kernel_size=(3,3),padding='same',activation='relu',activity_regularizer='l2'))
model.add(layers.MaxPool2D(pool_size=(2,2),padding='valid'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(120,activation='relu',activity_regularizer='l2'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_data,train_labels, epochs=10, batch_size=100,validation_data=(val_img,val_labels))



