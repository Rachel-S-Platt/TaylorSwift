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

from PIL import Image
from skimage import io
import dlib

import pygal
from pygal.style import Style

# Alex's Tensorflow code
########################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = str(os.getcwd()) + '/Taylorswiftboyfriends/'
IMAGE_SIZE = 64
BF_DIRECTORIES = ["Calvinharris", "Conor Kennedy", "HarryStyles", "Jake Gyllenhaal", "Joe Jonas", "John Mayer", "Taylor Lautner", "TomHiddleston"]
STEPS = 1000
DIR = './'
BF_IMAGE_DIR = "./static/img/Boyfriends/"

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, SESS_PATH)

def run_the_code(pass_x):
    array = sess.run(y_conv, feed_dict={x: pass_x, keep_prob: 1.0})
    array = array[0]

    min_val = np.min(array)
    if min_val < 0:
        min_val = - min_val
    array = array + min_val
    array = array / np.sum(array)

    results = sorted(zip(array, BF_DIRECTORIES), reverse=True)[:8]
    return (results)



########### END #############


######## FACE BLEND ########

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def combine_faces(name1, name2, name3):
    layer1 = io.imread(name1)
    detect1 = detect_faces(layer1)

    for n, face_rect in enumerate(detect1):
        layer1 = Image.fromarray(layer1).crop(face_rect)
        layer1.save("Test.png")

    layer2 = io.imread(name2)
    detect2 = detect_faces(layer2)

    for n, face_rect in enumerate(detect2):
        layer2 = Image.fromarray(layer2).crop(face_rect)

    layer3 = io.imread(name3)
    detect3 = detect_faces(layer3)

    for n, face_rect in enumerate(detect3):
        layer3 = Image.fromarray(layer3).crop(face_rect)


    layer1 = layer1.convert("RGBA")
    layer2 = layer2.convert("RGBA")
    layer3 = layer3.convert("RGBA")

    layer1 = layer1.resize((300, 300), Image.BILINEAR)
    layer2 = layer2.resize((300, 300), Image.BILINEAR)
    layer3 = layer3.resize((300, 300), Image.BILINEAR)

    new_img = Image.blend(layer1, layer2, 0.5)
    new_img2 = Image.blend(new_img, layer3, 0.5)
    new_img2.save("./static/img/Overlay/Blend.png","PNG")

def get_face(name):
    face = cv2.imread(name)
    b,g,r = cv2.split(face)           # get b, g, r
    face = cv2.merge([r,g,b])
    face2 = Image.fromarray(face)
    face2.save("Rachl.jpg")

    detect = detect_faces(face)
    face = Image.fromarray(face)
    face = face.crop(detect[0])
    face = face.resize((140, 140), Image.BILINEAR)

    img = Image.open('tyalor-tom.jpg')
    img_w, img_h = face.size
    offset = (img_w-20, img_h / 6)
    img.paste(face, offset)
    img.save("./static/img/Shopped/New.jpg")


########## MAPS ###########

song_map = {
    "Calvinharris": 3,
    "Conor Kennedy": 7,
    "HarryStyles": 2,
    "Jake Gyllenhaal": 0,
    "Joe Jonas": 6,
    "John Mayer": 1,
    "Taylor Lautner": 5,
    "TomHiddleston": 4
}

length_map = {
    "Calvinharris": 15,
    "Conor Kennedy": 3,
    "HarryStyles": 1,
    "Jake Gyllenhaal": 2,
    "Joe Jonas": 2,
    "John Mayer": 2,
    "Taylor Lautner": 4,
    "TomHiddleston": 3
}

image_map = {
    "Calvinharris": "CalvinHarris.jpg",
    "Conor Kennedy": "ConorKennedy.jpg",
    "HarryStyles": "HarryStyles.jpg",
    "Jake Gyllenhaal": "JakeGyllanhaal.jpg",
    "Joe Jonas": "JoeJonas.jpg",
    "John Mayer": "JohnMayer.jpg",
    "Taylor Lautner": "TaylorLautner.jpg",
    "TomHiddleston": "TomHiddleston.jpg"
}


######## ROUTES ########
music_dir = './static/music'

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
str1='Wazzzzzupp'

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    print(os.listdir(music_dir))
    song_to_play = -1
    blend = False
    shopped = False
    graph_gen= False
    music_files = [f for f in os.listdir(music_dir) if f.endswith('mp3')]
    r0=""
    r1=""
    r2=""
    r3=""
    r4=""
    r5=""
    r6=""
    r7=""

    boyfriend_result = ""
    length_of_rel=""
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

        r0=results[0][1] + ": " + str(results[0][0])
        r1=results[1][1] + ": " + str(results[1][0])
        r2=results[2][1] + ": " + str(results[2][0])
        r3=results[3][1] + ": " + str(results[3][0])
        r4=results[4][1] + ": " + str(results[4][0])
        r5=results[5][1] + ": " + str(results[5][0])
        r6=results[6][1] + ": " + str(results[6][0])
        r7=results[7][1] + ": " + str(results[7][0])

        print (results)

        length_of_rel= str(length_map[results[0][1]]*results[0][0]+length_map[results[1][1]]*results[1][0]+ length_map[results[2][1]]*results[2][0] + length_map[results[3][1]]*results[3][0] + length_map[results[4][1]]*results[4][0] + length_map[results[5][1]]*results[5][0] + length_map[results[6][1]]*results[6][0])
        song_to_play = song_map[boyfriend_result]

        bf_image1 = BF_IMAGE_DIR + image_map[results[0][1]]
        bf_image2 = BF_IMAGE_DIR + image_map[results[1][1]]
        bf_image3 = BF_IMAGE_DIR + image_map[results[2][1]]

        get_face("./static/img/" + filename)
        combine_faces(bf_image2, bf_image3, bf_image1)
        graphgen(results)
        graph_gen=True
        blend = True
        shopped = True

        print(results)

    return render_template('index.html', boyfriend_result=boyfriend_result, song_to_play=song_to_play, r0=r0, r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7, length_of_rel=length_of_rel, blend=blend, shopped=shopped)




## make a graph

custom_style = Style(
  background='transparent',
  plot_background='transparent',
  foreground='#000000',
  foreground_strong='#000000',
  foreground_subtle='#630C0D',
  opacity='.6',
  opacity_hover='.9',
  transition='400ms ease-in',
  colors=('#E853A0', '#E8537A', '#E95355', '#E87653', '#E89B53'))


def graphgen(results):

    line_chart = pygal.Bar(style=custom_style, show_legend=False)
    line_chart.title = 'Similarity rankings'

    line_chart.x_labels = results[0][1], results[1][1], results[2][1], results[3][1], results[4][1], results[5][1], results[6][1], results[7][1]

    line_chart.add('Similarity', [results[0][0], results[1][0], results[2][0], results[3][0], results[4][0], results[5][0], results[6][0], results[7][0]])
    line_chart.render_to_file('./static/img/chart.svg')
#    line_chart = line_chart.render_data_uri()
#    return render_template('index.html', line_chart=line_chart)
#



@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('index.html', str1=str1)

if __name__ == '__main__':
    app.run()
