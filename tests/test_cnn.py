import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow.contrib.layers as layers
import numpy as np

DIR = '/tmp/tensorflow/mnist/input_data'
NB_OUTPUT = 10
IMG_SIZE = 28
NB_PIXELS = IMG_SIZE**2
NB_ITERATIONS = 10000
BATCH_SIZE = 100
GRADIENT_STEP = 0.1
REG = 0.003
INIT_STD = 0.1

H_UNITS = 256
PATCH_SIZE = 3
CHANNELS = 16

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def main(train):
    # Import data
    mnist = input_data.read_data_sets(DIR, one_hot=True)

    # Input
    x = tf.placeholder(tf.float32, [None, NB_PIXELS])
    # x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, 1])
    y_true = tf.placeholder(tf.float32, [None, NB_OUTPUT])

    input_size = 784
    channels = 16
    img_size = int(np.sqrt(input_size))
    flat_size = int(img_size / 4) ** 2
    x_image = tf.reshape(x, [-1, img_size, img_size, 1])
    conv1 = layers.conv2d(x_image, channels, [3, 3], scope='conv1', activation_fn=tf.nn.relu)
    pool1 = layers.max_pool2d(conv1, [2, 2], scope='pool1')
    conv2 = layers.conv2d(pool1, channels, [3, 3], scope='conv2', activation_fn=tf.nn.relu)
    pool2 = layers.max_pool2d(conv2, [2, 2], scope='pool2')
    inputs = tf.reshape(pool2, [-1, flat_size * channels])

    # # Model
    # W_conv1 = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, 1, CHANNELS], stddev=INIT_STD))
    # b_conv1 = tf.Variable(tf.zeros([CHANNELS]))
    # h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    # h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # W_conv2 = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, CHANNELS, CHANNELS], stddev=INIT_STD))
    # b_conv2 = tf.Variable(tf.zeros([CHANNELS]))
    # h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    # h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # h_pool2_flat = tf.reshape(h_pool2, [-1, int(IMG_SIZE / 4)**2 * CHANNELS])
    fc_weights = tf.Variable(tf.truncated_normal([int(IMG_SIZE / 4)**2 * CHANNELS, H_UNITS], stddev=INIT_STD))
    fc_biases = tf.Variable(tf.zeros([H_UNITS]))
    fc = tf.nn.relu(tf.matmul(inputs, fc_weights) + fc_biases)

    weights = tf.Variable(tf.truncated_normal([H_UNITS, NB_OUTPUT], stddev=INIT_STD))
    biases = tf.Variable(tf.zeros([NB_OUTPUT]))
    y_pred = tf.matmul(fc, weights) + biases

    # Loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(GRADIENT_STEP).minimize(cross_entropy)

    # Run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    get_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Train
    if train:
        for i in range(NB_ITERATIONS):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={
                x: batch_xs,
                y_true: batch_ys})

            if not i % 100:
                train_accuracy = sess.run(
                    get_accuracy,
                    feed_dict={
                        x: mnist.train.images,
                        y_true: mnist.train.labels})

                test_accuracy = sess.run(
                    get_accuracy,
                    feed_dict={
                        x: mnist.test.images,
                        y_true: mnist.test.labels})
                print(i, train_accuracy, test_accuracy)

        train_accuracy = sess.run(
            get_accuracy, feed_dict={
                x: mnist.train.images,
                y_true: mnist.train.labels})

    # Test
    test_accuracy = sess.run(
        get_accuracy,
        feed_dict={
            x: mnist.test.images,
            y_true: mnist.test.labels})

    y_test_pred = sess.run(tf.argmax(y_pred, 1), {x: mnist.test.images})
    y_test_true = sess.run(tf.argmax(y_true, 1), {y_true: mnist.test.labels})

    print('Test accuracy: ', test_accuracy)
    print(confusion_matrix(y_test_true, y_test_pred))

    sess.close()

if __name__ == '__main__':
    main(train=True)

