import argparse

from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator

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
epochs = 1000
batch_size = 8
learning_rate = 0.1
n_channels = 3
n_classes = sum(1 for line in open(args.files + '/labels.txt'))  # total classes
dropout_rate = 0.5  # dropout, probability to keep units (while training)

nb_train_samples = sum(1 for line in open(train_file))
nb_validation_samples = sum(1 for line in open(test_file))
print("num images to train: " + str(nb_train_samples))
print("num images to test: " + str(nb_validation_samples))

if K.image_data_format() == 'channels_first':
    input_shape = (n_channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, n_channels)

# Network
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_initializer='truncated_normal'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu',kernel_initializer='truncated_normal'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='truncated_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))


adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# Other options:rotation_range, height_shift_range,featurewise_center, vertical_flip, featurewise_std_normalization...

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height),
                                                    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(img_width, img_height),
                                                        batch_size=batch_size, class_mode='categorical')
try:
    model.load_weights('my_model_weights.h5')
    '''
    model.load_weights('my_model_weights.h5', by_name=True)
    because you can set a name for every layer and only load the coincidenced layers by name
    model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
    '''
except:
    pass

checkpoint = ModelCheckpoint(filepath="./weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
                    validation_data=validation_generator, validation_steps=nb_validation_samples // batch_size,
                    callbacks=[checkpoint])

'''
mirar a ver el ejemplo que es x:test y que es y test
'''
score = model.evaluate_generator(validation_generator, nb_validation_samples / 5)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save_weights('my_model_weights.h5')