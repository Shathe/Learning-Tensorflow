# Learning-Tensorflow

The main aim of this project is to learn tensorflow framework. I will start almost from scratch, I am taking the MNIST example (edited by me adding some other stuff) and I will work on it.

First of all, I will set some steps/aims (related to the stuff I want to learn). As I complete the goals, I will be reporting the outcomes as well as the git commits which contains the new code.


- [x] Create an script to automatize the image collection process. For example given 10 lables to classify, get images from google searches and save the data-set ready to train
- [x] Choose what to Classify
- [ ] Load the data. Whether with the [tensorflow API](https://www.tensorflow.org/programmers_guide/reading_data) or just create a function which loads a random image/batch from a random labels, inside the main training loop, see [this example](https://github.com/asabater94/exeinos-uCode-2017/tree/master/ML), or numpy, or [DIY](http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels) loading the filepath and label and just [reading](http://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array ) the batch when needed with a nextBatch funtion
- [ ] Save and load the model in order to finetune and not to loose the training.
- [ ] Make a test script which gives both the predicted label and the probability.
- [ ] See the differences between adamOpt & GradientDescentOptimizer with lr decay
- [ ] Create a parallel version with either [Keras](https://keras.io/) or [Tflearn](http://tflearn.org/)
- [ ] Do a pre-processing of the data: Own cropping (depending the needs of the data set), data augmentation, mean substraction...
- [ ] Change the different layers (see other CNN's like VGG, imagenet, alexnet, Inception-v3, cifar10...) or try other [specific solutions] (http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030)
- [ ] Change the network structure (like siamese)
- [ ] Try with some well-known CNN and finetuning
- [ ] [Specify the gpu to use inside the code](https://www.tensorflow.org/tutorials/using_gpu)
- [ ] Show some filters, like [first conv filters](http://stackoverflow.com/questions/35759220/how-to-visualize-learned-filters-on-tensorflow)
- [ ] Make a distributed version

## Image colletor. Scrapping the web. Automatizing the process
The collecting data system is being automatized. The data is collected via google searches and is goint to be saved in a test & train directory as well as in one folder per each label. Each line ofthe [lables files](https://github.com/Shathe/Learning-Tensorflow/blob/master/labels.txt) is going to be a label to classify. In each line are more than one possible query, each one separated by comas.

In order to download all the images this command has to be executed:
```
python getData.py --dataFolder data # data is the folder to download the images
```
Then you can resize all the images just like this:
```
python resizeImages.py --dataFolder data --width 360 --heigh 320
```
## Choose what to Classify
First of all, I will begin to classify [places](https://github.com/Shathe/Learning-Tensorflow/blob/master/labels.txt):
* beach
* forest
* mountain
* city street
* village street
* building inside

Once I can classify those landscapse, I am thinking to classify something more useful such as some car models (for instance 10 models of Audi).