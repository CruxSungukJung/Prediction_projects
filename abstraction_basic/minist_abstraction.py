import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import sys
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
x_data,y_data = mnist.train.images,mnist.train.labels.astype(np.int32)
x_test,y_test = mnist.test.images,mnist.test.labels.astype(np.int32)

NUM_STEP = 2000
MINI_BATCH = 128

feature_col = learn.infer_real_valued_columns_from_input(x_data)

dnn = learn.DNNClassifier(
        feature_columns=feature_col,
        hidden_units=[200],
        n_classes=10,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.05))

dnn.fit(x=x_data,y=y_data,steps=NUM_STEP,batch_size=MINI_BATCH)

test_acc = dnn.evaluate(x=x_test,y=y_test,steps=1)["accuracy"]
print('{}'.format(test_acc))

