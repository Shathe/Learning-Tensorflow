# -*- coding: utf-8 -*-

from __future__ import division, absolute_import

import argparse

import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_utils import image_preloader
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from scipy import misc
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--imagePath", help="path of the image to predict")
parser.add_argument("--files", help="folder where the training files are placed (labels.txt, train.txt, test.txt...)")
parser.add_argument("--width", help="width for the images to resize")
parser.add_argument("--height", help="height for the images to resize")
args = parser.parse_args()

# Configuration variables
train_file = args.files + '/train.txt'
test_file = args.files + '/test.txt'
n_channels = 3
width = int(args.width)
height = int(args.height)

# Parameters
n_classes = sum(1 for line in open(args.files + '/labels.txt'))  # total classes (MNIST: 0-9 digits)
batch_size = 32
display_step = 100
learning_rate = 0.0001
dropout_rate = 0.5  # dropout, probability to keep units (while training)
n_epochs = 5
snapshot_steps = 2000


print("num images to train: " + str(sum(1 for line in open(train_file))))
print("num images to test: " + str(sum(1 for line in open(test_file))))

X, Y = image_preloader(train_file, image_shape=(width, height, n_channels), mode='file', categorical_labels=True, normalize=True)
testX, testY = image_preloader(test_file, image_shape=(width, height, n_channels), mode='file', categorical_labels=True, normalize=True)

'''
You can now apply some utils operations such us: image augmentation which is actually done in this example.
Sometimes there 'shuffle(X, Y)' because you can specify it in the model.fit function, or the to_categorical() function
which converts categorical numerical vectors to one_shot, but we have specified it in the image_preloader function
'''

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
# img_aug.add_random_rotation(max_angle=25.)


# Network
network = input_data(shape=[None, height, width, n_channels], data_preprocessing=img_prep, data_augmentation=img_aug)

# 32 filters, 5  filter_size,
network = conv_2d(network, 32, 5, activation='relu', weights_init='truncated_normal', bias_init='truncated_normal')
network = max_pool_2d(network, 2, strides=2)
network = conv_2d(network, 64, 5, strides=2, activation='relu', weights_init='truncated_normal',
                  bias_init='truncated_normal')
network = conv_2d(network, 64, 5, strides=2, activation='relu', weights_init='truncated_normal',
                  bias_init='truncated_normal')
network = max_pool_2d(network, 2, strides=2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, dropout_rate)
network = fully_connected(network, n_classes, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=learning_rate)

# Training
model = tflearn.DNN(network, checkpoint_path='model.tfl.ckpt', tensorboard_verbose=0, max_checkpoints=1)

model.load("model.tfl")

print("Total and final accuracy test: " + str(model.evaluate(testX, testY)[0]))

# Predicts what an image is given its path
def predict_image(image_path):
    image = misc.imread(image_path)
    image = misc.imresize(image, (height, width, n_channels))
    image = np.reshape(image, (1, height, width, n_channels))
    # Pick the probability and label of the first and only image.
    probs_img = model.predict(image.astype('float64'))[0]
    labels_img = model.predict_label(image.astype('float64'))[0]
    # As the result is given order by probability there is no need to use argmax, only to pick the ferst element
    prob = probs_img[0]
    label = labels_img[0]
    # Read the labels file in order to know what the label number stands for
    with open(args.files + '/labels.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    label = content[label].split(',')[0]

    print("Prediction:  " + str(label) + " with probability: " + str(prob * 100))

# Test given an image
predict_image(args.imagePath)




