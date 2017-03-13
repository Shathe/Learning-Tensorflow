# Learning-Tensorflow

The main aim of this project is to learn tensorflow framework. I will start almost from scratch, I am taking the MNIST example (edited by me adding some other stuff) and I will work on it.

First of all, I will set some steps/aims (related to the stuff I want to learn). As I complete the goals, I will be reporting the outcomes as well as the git commits which contains the new code.

- [ ] Create an script to automatize the image collection process. For example given 10 lables to classify, get images from google searches and save the data-set ready to train
- [ ] Choose what do I want to Classify
- [ ] Load the data. Whether with the [tensorflow API](https://www.tensorflow.org/programmers_guide/reading_data) or numpy, or [DIY](http://stackoverflow.com/questions/34340489/tensorflow-read-images-with-labels) loading the filepath and label and just [reading](http://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array ) the batch when needed with a nextBatch funtion
- [ ] Create a parallel version with either [Keras](https://keras.io/) or [Tflearn](http://tflearn.org/)
- [ ] Do a pre-processing of the data: Own cropping (depending the needs of the data set), data augmentation, mean substraction...
- [ ] Change the different layers (see other CNN's like VGG, imagenet, alexnet, Inception-v3, cifar10...) or try other [specific solutions] (http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030)
- [ ] Change the network structure (like siamese)
- [ ] Try with some well-known CNN and finetuning
- [ ] [Specify the gpu to use inside the code](https://www.tensorflow.org/tutorials/using_gpu)
- [ ] Show some filters, like [first conv filters] (http://stackoverflow.com/questions/35759220/how-to-visualize-learned-filters-on-tensorflow)
- [ ] Make a distributed version

## Image colletor. Scrapping the web. Automatizing the process

