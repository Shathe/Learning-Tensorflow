import argparse

from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy import misc
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--files", help="folder where the training files are placed (labels.txt, train.txt, test.txt...)")
parser.add_argument("--width", help="width for the images to resize")
parser.add_argument("--height", help="height for the images to resize")
args = parser.parse_args()

# Configuration variables
img_width, img_height = int(args.width), int(args.height)
train_data_dir = args.dataFolder + '/train'
validation_data_dir = args.dataFolder + '/test'
train_file = args.files + '/train.txt'
test_file = args.files + '/test.txt'
epochs = 30
batch_size = 16
learning_rate = 0.00001
n_channels = 3
n_classes = sum(1 for line in open(args.files + '/labels.txt'))  # total classes
dropout_rate = 0.5  # dropout, probability to keep units (while training)

nb_train_samples = sum(1 for line in open(train_file))
nb_validation_samples = sum(1 for line in open(test_file))
print("num images to train: " + str(nb_train_samples))
print("num images to test: " + str(nb_validation_samples))

if K.image_data_format() == 'channels_first':
    #Theano backend
    input_shape = (n_channels, img_width, img_height)
else:
    #Tensorflow backend
    input_shape = (img_width, img_height, n_channels)


def get_label(dict, index):
    for label, index_dict in dict.iteritems():
        if index == index_dict:
            return label

# Network
model = Sequential()
model.add(Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=input_shape,
                 kernel_initializer='truncated_normal'))
model.add(Conv2D(32, (5, 5), padding='same', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (5, 5), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(n_classes))
model.add(Activation('softmax'))
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, rotation_range=90)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),

                                                        batch_size=batch_size, class_mode='categorical', shuffle=True)
try:
    model.load_weights('weights.hdf5')
    print("Weights loaded")
    '''
    model.load_weights('my_model_weights.h5', by_name=True)
    because you can set a name for every layer and only load the coincidenced layers by name
    model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
    '''
except:
    pass

checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpoint])

score = model.evaluate_generator(validation_generator, nb_validation_samples)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save_weights('weights_final.hdf5')

# Predicts what an iamge is given its path
def predict_image(image_path):
    # Read the image, rescale and reshape it (as the neural network expects it)
    image = misc.imread(image_path)
    image = misc.imresize(image, input_shape)
    image = np.reshape(image, (1, img_width, img_height, n_channels))
    print(model.predict(image))
    prediction = model.predict(image)
    # Prediction of the first and the only image predicted
    prediction = prediction[0]
    index = np.argmax(prediction)
    prob = prediction[index]
    label = get_label(validation_generator.class_indices, index)
    print("Prediction:  " + str(label) + " with probability: " + str(prob * 100))

# Test given an image
predict_image(validation_data_dir + '/mountain/_mountain_landscape_131.jpg')

