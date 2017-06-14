import argparse

from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace
from scipy import misc
import math
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--files", help="folder where the training files are placed (labels.txt, train.txt, test.txt...)")
parser.add_argument("--weights", help="height for the images to resize")
parser.add_argument("--imagePath", help="path of the image to predict")

args = parser.parse_args()
train_data_dir = args.dataFolder + '/train'
validation_data_dir = args.dataFolder + '/test'
train_file = args.files + '/train.txt'
test_file = args.files + '/test.txt'
nb_train_samples = sum(1 for line in open(train_file))
nb_validation_samples = sum(1 for line in open(test_file))
epochs = 5
batch_size = 8
learning_rate = 0.00001
n_classes = 0

for _, dirnames, _ in os.walk(train_data_dir):
  # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)

# create the base pre-trained model

hidden_dim = 512

image_input = Input(shape=(224, 224, 3))
# for theano uncomment
# image_input = Input(shape=(3,224, 224))
base_model = VGGFace(input_tensor=image_input, include_top=False)
last_layer = base_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(n_classes, activation='softmax', name='fc8')(x)
model = Model(image_input, out)


def preprocess_img(img):
    img = img.astype(np.float32) / 255.0
    img -= 0.5
    return img * 2

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=45, preprocessing_function=preprocess_img)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(224, 224),
                                                        batch_size=(batch_size/2), class_mode='categorical', shuffle=True)



# Load weights
try:
    print(model.load_weights(args.weights))
    print("Weights loaded")
except:
    pass


# get the label given the index
def get_label(dict, index):
    for label, index_dict in dict.iteritems():
        if index == index_dict:
            return label


# Predicts what an image is given its path
def predict_image(image_path):
    # Read the image, rescale and reshape it (as the neural network expects it)
    image = misc.imread(image_path)
    image = misc.imresize(image, (224, 224, 3))
    image = preprocess_img(image)
    image = np.reshape(image, (1, 224, 224, 3))
    prediction = model.predict(image)
    # Prediction of the first and the only image predicted
    prediction = prediction[0]
    index = np.argmax(prediction)
    prob = float(prediction[index])
    label = get_label(validation_generator.class_indices, index)
    print("Prediction: " + str(label) + " with probability: " + str(prob * 100.0))
    if prob < 0.6/math.log10(n_classes + 2 ):
        if prob < 0.25 / math.log10(n_classes + 2):
            print("DESCONOCIDO")
        else:
            print("PROBABLEMENTE DESCONOCIDO")


# Test given an image
predict_image(args.imagePath)
