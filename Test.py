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
IMAGE_SIZE = 200
BF_DIRECTORIES = ["Calvinharris", "Conor Kennedy", "HarryStyles", "Jake Gyllenhaal", "Joe Jonas", "John Mayer", "Taylor Lautner", "TomHiddleston"]
STEPS = 100

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)



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
                    images = np.array([image])
                    distorted = seq.augment_images(images)
                    print(distorted.shape)
                    print(image.shape)
                    print(distorted)
                    self.images.append(distorted[0])
                    self.images.append(image)
                    self.labels.append(self.one_hot(boyfriend))

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.images = self.images.astype('float32')
        self.images = np.multiply(self.images, 1.0/255.0)

    # https://medium.com/@tifa2up/image-classification-using-deep-neural-networks-a-beginner-friendly-approach-using-tensorflow-94b0a090ccd4
    def pre_process_image(self, image):
        image = tf.random_crop(image, size=[IMAGE_SIZE, IMAGE_SIZE, 3])
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

        return image

    def one_hot(self, boyfriend, vals=8):
        out = np.zeros(8)
        out[BF_DIRECTORIES.index(boyfriend)] = 1
        return out

    def get_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=.2)
        return X_train, X_test, y_train, y_test

    def check_shape(self):
        print(self.images.shape)

image_loader = ImageLoader()
image_loader.load_images()
print(image_loader.images.shape)

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)
