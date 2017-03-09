﻿# Learning-Tensorflow

The main aim of this project is to learn tensorflow framework. I will start almost from scratch, I am taking the MNIST example (edited by me adding some other stuff) and I will work on it.

First of all, I will set some steps/aims (related to the stuff I want to learn). As I complete the goals, I will be reporting the outcomes as well as the git commits which contains the new code.

- [x] Choose the data set to work on.
- [ ] Create a pararell version with either [Keras](https://keras.io/) or [Tflearn](http://tflearn.org/)
- [ ] Load the data. Whether with the [tensorflow API](https://www.tensorflow.org/programmers_guide/reading_data) or numpy, or [DIY](http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels) loading the filepath and label and just [reading](http://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array ) the batch when needed with a nextBatch funtion
- [ ] Do a preprocessing of the data: Own cropping (depending the needs of the data set), data augmentation...
- [ ] Change the different layers (see other CNN's like VGG, imagenet, alexnet, Inception-v3, cifar10...)
- [ ] Change the network structure (like siamese)
- [ ] Try with some well-known CNN and finetuning it
- [ ] [Specify the gpu to use inside the code](https://www.tensorflow.org/tutorials/using_gpu)
- [ ] Show some filters, like [first conv filters] (http://stackoverflow.com/questions/35759220/how-to-visualize-learned-filters-on-tensorflow)
- [ ] Make a distributed version

## Choose the data set to work on

The selected [data-set](https://www.cs.toronto.edu/~kriz/cifar.html) is the [Cifar-10](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (As Cifar-100 has the same type of images, it will be also used).

Speaking about Cifar-10, Classifying the images randomly, the obtained average accuracy would be around 10%. On the other hand, the pro's results shows a 96% of accuracy on the test data-set can be reached, while on the Cifar-100, a 75% is also accomplishable.
As I am learning, I may set my hopes on getting 50% (for example :D ) and as for the Cifar-100, 20%. I am just guessing.