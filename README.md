# Learning-Tensorflow

The main aim of this project is to learn tensorflow framework. I will start almost from scratch, I am taking the MNIST example (edited by me adding some other stuff) and I will work on it.

First of all, I will set some steps/aims (related to the stuff I want to learn). As I complete the goals, I will be reporting the outcomes as well as the git commits which contains the new code.


- [x] Create an script to automatize the image collection process. For example given 10 lables to classify, get images from google searches and save the data-set ready to train
- [x] Choose what to Classify
- [x] Load the data.
- [x] Have 2 versions: Keras & TFLearn
- [ ] Save and load the model in order to finetune and not to loose the training.
- [ ] Predict. Make a test script which gives both the predicted label and the probability.
- [ ] Do image augmentation with the API's
- [ ] Change the different layers (see other CNN's like VGG, imagenet, alexnet, Inception-v3, cifar10...) [keras](https://github.com/fchollet/deep-learning-models) or [tflearn](https://github.com/tflearn/tflearn/tree/master/examples/images)
- [ ] Try to set up a siamesse network [TFLearn](https://github.com/tflearn/tflearn/issues/544) and[Keras](https://github.com/fchollet/keras/blob/master/examples/mnist_siamese_graph.py)
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
python resizeImages.py --dataFolder data --width 160 --heigh 120
```

## Keras and TFLearn models.

I code both of them but I am going to work only with Keras because it gives more configurability, it is more used and updated.

In order to execute it code:
```
python model_keras.py --dataFolder data --files  . --width 160 --height 120
```