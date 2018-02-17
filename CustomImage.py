import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import random
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = str(os.getcwd()) + '/Taylorswiftboyfriends/'
IMAGE_SIZE = 64
BF_DIRECTORIES = ["Calvinharris", "Conor Kennedy", "HarryStyles", "Jake Gyllenhaal", "Joe Jonas", "John Mayer", "Taylor Lautner", "TomHiddleston"]
STEPS = 1000
DIR = './'

full_path =  './Alex.jpg'
image = cv2.imread(full_path)
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR)
image = np.array(image)
image = image.astype('float32')
# print(image.shape)

iteration = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial, name="bias")

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


tf.reset_default_graph()

META_PATH = "./new2000/model_cpkt2000.meta"
SESS_PATH = "./new2000/model_cpkt2000"

x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="placeholder_x")
y_ = tf.placeholder(tf.float32, shape=[None, 8], name="placeholder_y")

new_x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
new_y = tf.placeholder(tf.float32, shape=[None, 8])

keep_prob = tf.placeholder(tf.float32, name="placeholder_keepprob")

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv3 = conv_layer(conv2_pool, shape=[5, 5, 64, 128])
conv3_pool = max_pool_2x2(conv3)
conv3_flat = tf.reshape(conv3_pool, [-1, 8*8*128])
conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 8)

saver = tf.train.Saver()

def run_the_code():
    with tf.Session() as sess:
        pass_x = np.array([image])
        # print(pass_x.shape)
        #
        # print(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        # saver = tf.train.import_meta_graph(META_PATH)
        saver.restore(sess, SESS_PATH)

        array = sess.run(y_conv, feed_dict={x: pass_x, keep_prob: 1.0})
        array = array[0]
        mean = np.sum(np.absolute(array))
        max_val = np.max(array)
        min_val = np.min(array)
        if min_val < 0:
            min_val = - min_val
        print(array)
        array = array + min_val
        print(array)
        array = array / np.sum(array)
        print(array)
        results = sorted(zip(array, BF_DIRECTORIES), reverse=True)[:8]
        # print(max_val)
        r1=results[0][1] + ": " + str(results[0][0])
        print(results[0][1] + ": " + str(results[0][0]))
        r2=results[1][1] + ": " + str(results[1][0])
        print(results[1][1] + ": " + str(results[1][0]))
        r3= results[2][1] + ": " + str(results[2][0])
        print(results[2][1] + ": " + str(results[2][0]))
        r4= results[3][1] + ": " + str(results[3][0])
        print(results[3][1] + ": " + str(results[3][0]))
        r5= results[4][1] + ": " + str(results[4][0])
        print(results[4][1] + ": " + str(results[4][0]))
        r6= results[5][1] + ": " + str(results[5][0])
        print(results[5][1] + ": " + str(results[5][0]))
        r7= results[6][1] + ": " + str(results[6][0])
        print(results[6][1] + ": " + str(results[6][0]))
        r8= results[7][1] + ": " + str(results[7][0])
        print(results[7][1] + ": " + str(results[7][0]))
    


    return results

run_the_code()
