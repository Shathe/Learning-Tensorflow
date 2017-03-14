'''
Author: Inigo Alonso ruiz
Date: 14-10-2016
Project: https://github.com/Shathe/tensor-flowjor/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Parameters
# learning_rate = 0.001
training_iters = 200000
batch_size = 50
display_step = 100
learning_rate = 0.001

# Network Parameters
n_input = 784  # data input (MNIST: img shape: 28*28)
n_classes = 10  # total classes (MNIST: 0-9 digits)
dropout = 0.5  # dropout, probability to keep units (while training)

sess = tf.InteractiveSession()

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


def conv2d(x, W, b, strides=1, padding='SAME', relu=True, bias=True):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    if bias:
        x = tf.nn.bias_add(x, b)
    if relu:
        x = tf.nn.relu(x)
    return x


def maxpool2d(x, k=2, padding='SAME'):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding=padding)


# Create model
def conv_net(x, weights, biases, dropout):
    # To apply the layer, we first reshape x to a 4d tensor,
    # with the second and third dimensions corresponding to input image width and height,
    # and the final dimension corresponding to the number of color channels.
    # 28 , 28 because the iamge is 28x28
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
  
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64])),

    # Now that the image size has been reduced to 7x7, (2 times maxpool applied 2x2)
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, n_classes]))

    # Another good function instead og random_normal is tf.truncated_normal

}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bd1': tf.Variable(tf.truncated_normal([1024])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
    # Another good function instead og random_normal is tf.constant(0.1, shape=shape)
}

#CoupledDecay learning rate
'''
batch = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
0.003, # Base learning rate.
batch * batch_size, # Current index into the dataset.
training_iters/50, # Decay step.
0.95, # Decay rate.
staircase=True)
'''


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Evaluate model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cost, global_step=batch)

# correct prediction
accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))



# Initializing the variables
sess.run(tf.global_variables_initializer())
step = 1
# Keep training until reach max iterations
while step * batch_size < training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Run optimization op (backprop)
    c = sess.run([optimizer], feed_dict={x: batch_x, y: batch_y,
                                   keep_prob: dropout})
    if step % display_step == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keep_prob: 1.})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
    # Write logs at every iteration
    step += 1
print("Optimization Finished!")

# Calculate accuracy for 1000 mnist test images
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: mnist.test.images[:1000],
                                  y: mnist.test.labels[:1000],
                                  keep_prob: 1.}))


