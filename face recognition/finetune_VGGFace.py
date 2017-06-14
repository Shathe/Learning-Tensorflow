import argparse
import os
import numpy as np
from keras import optimizers
from keras.engine import Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace

parser = argparse.ArgumentParser()
parser.add_argument("--dataFolder", help="folder where the images are going to be saved")
parser.add_argument("--files", help="folder where the training files are placed (labels.txt, train.txt, test.txt...)")

args = parser.parse_args()
train_data_dir = args.dataFolder + '/train'
validation_data_dir = args.dataFolder + '/test'
train_file = args.files + '/train.txt'
test_file = args.files + '/test.txt'
nb_train_samples = sum(1 for line in open(train_file))
nb_validation_samples = sum(1 for line in open(test_file))
epochs = 8
batch_size = 32
learning_rate = 0.00001
n_classes = 0

for _, dirnames, _ in os.walk(train_data_dir):
    # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)

print("n_classes: " + str(n_classes))
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

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
adam = optimizers.adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


def preprocess_img(img):
    img = img.astype(np.float32) / 255.0
    img -= 0.5
    return img * 2



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(shear_range=0.25, width_shift_range=0.3, height_shift_range=0.3,
                                   zoom_range=0.4, rotation_range=0.3,  preprocessing_function=preprocess_img)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)



# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(224, 224),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(224, 224),
                                                        batch_size=(batch_size / 2), class_mode='categorical',
                                                        shuffle=True)

# train the model on the new data for a few epochs


model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)


score = model.evaluate_generator(validation_generator, nb_validation_samples)
model.save( 'weights2.h5')
# model.save('train_255.h5')
# testear  el primero nrmal y el segudno normal y quitando media
print('Test loss:', score[0])
print('Test accuracy:', score[1])
