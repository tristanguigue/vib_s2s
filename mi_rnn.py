import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

DIR = '/tmp/tensorflow/mnist/input_data'
H1_UNITS = 100
START_SEQ = 0
SEQ_SIZE = 28 * 28
BOTTLENECK_SIZE = 32
LSTM_SIZE = 128
ENCODER_OUTPUT = 2 * BOTTLENECK_SIZE
DECODER_OUTPUT = 1
NB_EPOCHS = 500
TRAIN_BATCH = 200
BETA = 0.001
BIT_SIZE = 1


def tf_binarize(images, threshold=0.1):
    return tf.cast(threshold < images, tf.float32)


def get_batch_accuracy_loss(data):
    accuracies = []
    nb_batches = data.shape[0] // TRAIN_BATCH
    for i in range(nb_batches):
        data_batch = data[i * TRAIN_BATCH: (i + 1) * TRAIN_BATCH, START_SEQ:START_SEQ + SEQ_SIZE]
        batch_accuracy = sess.run(get_accuracy, feed_dict={x: data_batch})
        accuracies.append(batch_accuracy)

    return np.mean(accuracies)


mnist = input_data.read_data_sets(DIR, one_hot=True)
standard_mu = tf.zeros(BOTTLENECK_SIZE)
standard_sigma = tf.ones(BOTTLENECK_SIZE)
multivariate_std = tf.contrib.distributions.MultivariateNormalDiag(standard_mu, standard_sigma)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, SEQ_SIZE], name='x-input')
    inputs = tf.expand_dims(tf_binarize(x), 2)

with tf.name_scope('encoder_h1'):
    stack = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE)
    outputs, state = tf.nn.dynamic_rnn(stack, inputs, dtype=tf.float32)
    flat_outputs = tf.reshape(outputs, [-1, LSTM_SIZE])

    out_weights = tf.Variable(
        tf.truncated_normal([LSTM_SIZE, ENCODER_OUTPUT], stddev=0.1),
        name='out_weights')
    out_biases = tf.Variable(
        tf.truncated_normal([ENCODER_OUTPUT], stddev=0.1),
        name='out_biases')
    encoder_output = tf.matmul(flat_outputs, out_weights) + out_biases


with tf.name_scope('sample'):
    mu = encoder_output[:, :BOTTLENECK_SIZE]
    sigma = tf.nn.softplus(encoder_output[:, BOTTLENECK_SIZE:])
    epsilon = tf.reshape(multivariate_std.sample(), [-1, 1])
    z = mu + tf.matmul(sigma, epsilon)

with tf.name_scope('decoder'):
    decoder_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_SIZE, DECODER_OUTPUT], stddev=0.001))
    decoder_biases = tf.Variable(tf.constant(0.001, shape=[DECODER_OUTPUT]))
    decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
    decoder_output = tf.reshape(decoder_output, [-1, SEQ_SIZE, DECODER_OUTPUT])

with tf.name_scope('loss'):
    true_pixels = inputs[:, 1:]
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=true_pixels, logits=decoder_output[:, :-1])

    kl = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)
    kl = tf.reshape(kl, [-1, SEQ_SIZE, DECODER_OUTPUT])

    loss = tf.reduce_mean(cross_entropy + BETA * kl[:, :-1])

# Optimiser
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# Test
with tf.name_scope('accuracy'):
    predicted_pixels = tf.round(tf.sigmoid(decoder_output[:, :-1]))
    correct_predictions = tf.equal(predicted_pixels, true_pixels)
    get_accuracy = 100 * tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Training
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

for epoch in range(NB_EPOCHS):
    epoch_batches = int(mnist.train.num_examples / TRAIN_BATCH)
    total_loss = 0

    for i in range(epoch_batches):
        batch_xs, _ = mnist.train.next_batch(TRAIN_BATCH)
        _, current_loss = sess.run([train_step, loss], feed_dict={
            x: batch_xs[:, START_SEQ:START_SEQ + SEQ_SIZE]})
        total_loss += current_loss

    print('\nEpoch:', epoch)

    train_accuracy = get_batch_accuracy_loss(mnist.train.images)
    test_accuracy = get_batch_accuracy_loss(mnist.test.images)

    print('Loss: ', total_loss / epoch_batches)
    print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)

sess.close()