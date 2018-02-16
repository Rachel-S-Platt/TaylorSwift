from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import random
from sklearn.model_selection import train_test_split
import imgaug as ia
from imgaug import augmenters as iaa

# Alex's Tensorflow code
########################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = str(os.getcwd()) + '/Taylorswiftboyfriends/'
IMAGE_SIZE = 64
BF_DIRECTORIES = ["Calvinharris", "Conor Kennedy", "HarryStyles", "Jake Gyllenhaal", "Joe Jonas", "John Mayer", "Taylor Lautner", "TomHiddleston"]
STEPS = 1000
DIR = './'

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

def run_the_code(pass_x):
    with tf.Session() as sess:
        # print(pass_x.shape)
        #
        # print(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        # saver = tf.train.import_meta_graph(META_PATH)
        saver.restore(sess, SESS_PATH)
        # print("\n\n")
        # print(tf.global_variables())
        # print(sess.run('weight:0'))



        # w1 = tf.get_default_graph().get_tensor_by_name("weight:0")
        # print(w1)

        array = sess.run(y_conv, feed_dict={x: pass_x, keep_prob: 1.0})
        array = array[0]
        results = sorted(zip(array, BF_DIRECTORIES), reverse=True)[:3]
        mean = np.sum(np.absolute(array))
        max_val = np.max(array)
        # print(max_val)
        print(results[0][1] + ": " + str(results[0][0] / mean))
        print(results[1][1] + ": " + str(results[1][0] / mean))
        print(results[2][1] + ": " + str(results[2][0] / mean))

        return results



########### END #############
#############################

music_dir = './static/music'

app = Flask(__name__)
str1='Wazzzzzupp'

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print(os.listdir(music_dir))
    song_to_play = -1
    music_files = [f for f in os.listdir(music_dir) if f.endswith('mp3')]
    music_files_number = len(music_files)
    boyfriend_result = ""
    print(music_files)
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        full_path =  './static/img/' + filename
        image = cv2.imread(full_path)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR)
        image = np.array(image)
        image = image.astype('float32')
        pass_x = np.array([image])

        results = run_the_code(pass_x)
        boyfriend_result = results[0][1]
        print(results)

        # return filename
    return render_template('index.html', music_files = music_files, music_files_number = music_files_number, song_to_play=song_to_play, boyfriend_result=boyfriend_result)

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('index.html', str1=str1)

if __name__ == '__main__':
    app.run()
