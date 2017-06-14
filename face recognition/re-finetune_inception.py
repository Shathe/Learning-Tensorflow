import argparse

from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

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

def pop(self):
    '''Removes a layer instance on top of the layer stack.
    '''
    if not self.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self.inbound_nodes = []
            self.outbound_nodes = []
        else:
            self.layers[-1].outbound_nodes = []
            self.outputs = [self.layers[-1].output]
        self.built = False


# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False)
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

model.load_weights("my_model_final2.h5", by_name=True)

# Quitamos ultiam capa y ponemos numero de clases que queremos
pop(model)
x = model.layers[-1].output

predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, width_shift_range=0.15, height_shift_range=0.15,
                                   zoom_range=0.4, rotation_range=45)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(299, 299),
                                                    batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(299, 299),
                                                        batch_size=(batch_size / 2), class_mode='categorical',
                                                        shuffle=True)

# train the model on the new data for a few epochs



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
