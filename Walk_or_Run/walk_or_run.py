import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,2])

run_label = 1
walk_lable = 0
file_address = 'train/run/'
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

test_img,test_label = _parse_function(file_address+'run_0a9284fb.png',run_label)


sess = tf.Session()

sess.run(tf.global_variables_initializer())
img = sess.run(test_img)
