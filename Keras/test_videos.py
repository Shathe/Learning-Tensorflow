import argparse

from keras import backend as K
from keras import optimizers
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--files", help="folder where the training files are placed (labels.txt, train.txt, test.txt...)")
parser.add_argument("--width", help="width for the images to resize")
parser.add_argument("--height", help="height for the images to resize")
parser.add_argument("--weights", help="height for the images to resize")
args = parser.parse_args()

# Configuration variables
img_width, img_height = int(args.width), int(args.height)
train_data_dir = args.dataFolder + '/train'
validation_data_dir = args.dataFolder + '/test'
train_file = args.files + '/train.txt'
test_file = args.files + '/test.txt'
epochs = 400
batch_size = 8
learning_rate = 0.00001
n_channels = 3
dropout_rate = 0.5  # dropout, probability to keep units (while training)
import os
n_classes = 0

for _, dirnames, _ in os.walk(train_data_dir):
  # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)


nb_train_samples = sum(1 for line in open(train_file))
nb_validation_samples = sum(1 for line in open(test_file))
print("num images to train: " + str(nb_train_samples))
print("num images to test: " + str(nb_validation_samples))

if K.image_data_format() == 'channels_first':
    # Theano backend
    input_shape = (n_channels, img_width, img_height)
else:
    # Tensorflow backend
    input_shape = (img_width, img_height, n_channels)

# Network
model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape,
                 kernel_initializer='truncated_normal'))
model.add(Conv2D(32, (5, 5), padding='same', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                 kernel_initializer='truncated_normal'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                   rotation_range=90)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='categorical', shuffle=True)

# Load weights
try:
    model.load_weights(args.weights)
    score = model.evaluate_generator(validation_generator, nb_validation_samples)
    score = model.evaluate_generator(validation_generator, nb_validation_samples)
    print('Test loss weights:', score[0])
    print('Test accuracy weights:', score[1])
except:
    pass
