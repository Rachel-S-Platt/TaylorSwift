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

PATH = str(os.getcwd()) + '/TaylorSwift/Taylorswiftboyfriends/'
IMAGE_SIZE = 64
BF_DIRECTORIES = ["Calvinharris", "Conor Kennedy", "HarryStyles", "Jake Gyllenhaal", "Joe Jonas", "John Mayer", "Taylor Lautner", "TomHiddleston"]
STEPS = 1000
DIR = './'

iteration = 1

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

# Class that was used in order to get all of the boyfriend images
class ImageLoader:
    def __init__(self):
        self.images = []
        self.labels = []

    def load_images(self):
        for boyfriend in BF_DIRECTORIES:
            path_to_directory = PATH + boyfriend
            for image_name in os.listdir(path_to_directory):
                if image_name != ".DS_Store":
                    # print(image_name)
                    full_path =  path_to_directory + "/" + image_name
                    image = cv2.imread(full_path)
                    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), cv2.INTER_LINEAR)
                    image = np.array(image)
                    image = image.astype('float32')
                    for i in range(5):
                        image_array = np.array([image])
                        distorted = seq.augment_images(image_array)
                        self.images.append(distorted[0])
                        self.labels.append(self.one_hot(boyfriend))
                    self.images.append(image)
                    self.labels.append(self.one_hot(boyfriend))

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(self.images.shape)
        print(self.labels.shape)
        self.images = self.images.astype('float32')
        self.images = np.multiply(self.images, 1.0/255.0)

    def one_hot(self, boyfriend, vals=8):
        out = np.zeros(8)
        out[BF_DIRECTORIES.index(boyfriend)] = 1
        return out

    def get_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=.2)
        return X_train, X_test, y_train, y_test

    def check_shape(self):
        print(self.images.shape)



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

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

print(tf.get_default_graph)
image_loader = ImageLoader()
# loader.check_shape()
image_loader.load_images()
X_batch, X_test, y_batch, y_test = image_loader.get_train_test()


X_batches = np.array_split(X_batch, 4)
y_batches = np.array_split(y_batch, 4)
X_batches = np.array([np.array(X_batches[0]), np.array(X_batches[1]), np.array(X_batches[2]), np.array(X_batches[3])])
y_batches = np.array([np.array(y_batches[0]), np.array(y_batches[1]), np.array(y_batches[2]), np.array(y_batches[3])])


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

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# def test(sess):
#     global iteration
#     global X_test
#     global y_test
#
#     X = X_test
#     Y = y_test
#     acc = sess.run(accuracy, feed_dict={x: X, y_: Y, keep_prob: 1.0})
#
#     print("Iteration: {}, Accuracy: {:.4}%".format(iteration, acc * 100))
#     iteration += 1



with tf.Session() as sess:
    X_test
    y_test

    X_for_feed = X_test
    Y_for_feed = y_test
    pass_x = np.array([X_test[0]])
    pass_y = np.array([y_test[0]])
    print("++++++++++")
    print(pass_x.shape)
    print(pass_y.shape)
    print(X_test.shape)
    print(y_test.shape)

    saver = tf.train.import_meta_graph(META_PATH)
    # saver = tf.train.Saver()
    saver.restore(sess, SESS_PATH)
    sess.run(tf.global_variables_initializer())
    # w1 = tf.get_variable("weight:0", shape=[5, 5, 3, 32])
    # print(w1.eval())
    # print(tf.global_variables())
    f = open("Variables.txt", "w")
    f.write(str(tf.global_variables()))
    f.close()

    acc = sess.run(accuracy, feed_dict={x: pass_x, y_: pass_y, keep_prob: 1.0})
    array = sess.run(y_conv, feed_dict={x: pass_x, y_: pass_y, keep_prob: 1.0})
    array = array[0]
    results = list(zip(BF_DIRECTORIES, array))
    print(results)

    print("Iteration: {}, Accuracy: {:.4}%".format(iteration, acc * 100))


    # for i in range(STEPS):
    #     # X_batch, X_test, y_batch, y_test = image_loader.get_train_test()
    #     batch = random.randint(0, 3)
    #     X_batch = X_batches[batch]
    #     y_batch = y_batches[batch]
    #
    #     zipper = list(zip(X_batch, y_batch))
    #     random.shuffle(zipper)
    #
    #     X_batch, y_batch = zip(*zipper)
    #     X_batch = np.array(X_batch)
    #     y_batch = np.array(y_batch)
    #
    #
    #     sess.run(train_step, feed_dict={x: X_batch, y_: y_batch, keep_prob: 0.5})
    #
    #     test(sess)
    #
    #     if i % 50 == 0:
    #         saver.save(sess, os.path.join(DIR, "new_model_cpkt" + str(i), "model_cpkt" + str(i)))
