import argparse

from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

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
epochs = 5
batch_size = 8
learning_rate = 0.00001

import os
n_classes = 0

for _, dirnames, _ in os.walk(train_data_dir):
  # ^ this idiom means "we won't be using this value"
    n_classes += len(dirnames)


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
# print(base_model.load_weights("my_model_final.h5", by_name=True))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 4 classes
predictions = Dense(n_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

adam = optimizers.adam(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.4, rotation_range=45)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(299, 299),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(299, 299),
                                                        batch_size=(batch_size/2), class_mode='categorical', shuffle=True)

# train the model on the new data for a few epochs


model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
'''
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
'''
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
adam = optimizers.adam(lr=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size)

score = model.evaluate_generator(validation_generator, nb_validation_samples)
model.save('my_model_final.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])


'''
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
'''
