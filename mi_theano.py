import theano as th
from theano import tensor
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

DIR = '/tmp/tensorflow/mnist/input_data'
H1_UNITS = 124
H2_UNITS = 124
IMG_SIZE = 28 * 28
BOTTLENECK_SIZE = 256
ENCODER_OUTPUT = 2 * BOTTLENECK_SIZE
DECODER_OUTPUT = 10
NB_EPOCHS = 500
TRAIN_BATCH = 200
BETA = 0.0001

mnist = input_data.read_data_sets(DIR, one_hot=True)
mvn = tensor.raw_random.RandomStreamsBase()

x = theano.shared()
y_true = tensor.shared()

h1_weights = tensor.fscalar()
h2_weights = tensor.fscalar()


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, IMG_SIZE], name='x-input')
    y_true = tf.placeholder(tf.float32, [None, DECODER_OUTPUT], name='y-input')

with tf.name_scope('encoder_h1'):
    h1_weights = tf.Variable(tf.truncated_normal([IMG_SIZE, H1_UNITS], stddev=0.001))
    h1_biases = tf.Variable(tf.constant(0.001, shape=[H1_UNITS]))
    hidden1 = tf.nn.relu(tf.matmul(x, h1_weights) + h1_biases)

with tf.name_scope('encoder_h2'):
    h2_weights = tf.Variable(tf.truncated_normal([H1_UNITS, H2_UNITS], stddev=0.001))
    h2_biases = tf.Variable(tf.constant(0.001, shape=[H2_UNITS]))
    hidden2 = tf.nn.relu(tf.matmul(hidden1, h2_weights) + h2_biases)

with tf.name_scope('encoder_out'):
    enc_out_weights = tf.Variable(tf.truncated_normal([H2_UNITS, ENCODER_OUTPUT], stddev=0.001))
    enc_out_biases = tf.Variable(tf.constant(0.001, shape=[ENCODER_OUTPUT]))
    encoder_output = tf.matmul(hidden2, enc_out_weights) + enc_out_biases

with tf.name_scope('sample'):
    mu = encoder_output[:, :BOTTLENECK_SIZE]
    sigma = tf.nn.softplus(encoder_output[:, BOTTLENECK_SIZE:])

    s = mvn.normal(size=(BOTTLENECK_SIZE), avg=0.0, std=1.0, ndim=None):
    epsilon = tf.reshape(multivariate_std.sample(), [-1, 1])
    z = mu + tf.matmul(sigma, epsilon)

with tf.name_scope('decoder'):
    decoder_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_SIZE, DECODER_OUTPUT], stddev=0.001))
    decoder_biases = tf.Variable(tf.constant(0.001, shape=[DECODER_OUTPUT]))
    decoder_output = tf.matmul(z, decoder_weights) + decoder_biases

with tf.name_scope('loss'):
    x_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=decoder_output)
    kl = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)
    loss = tf.reduce_mean(x_ent + BETA * kl)

# Optimiser
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Test
with tf.name_scope('accuracy'):
    correct_predictions = tf.equal(tf.arg_max(decoder_output, 1), tf.argmax(y_true, 1))
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
        batch_xs, batch_ys = mnist.train.next_batch(TRAIN_BATCH)

        _, current_loss = sess.run([train_step, loss], feed_dict={
            x: batch_xs,
            y_true: batch_ys})

        total_loss += current_loss

    print('\nEpoch:', epoch)

    train_accuracy = get_batch_accuracy_loss(mnist.train.images, mnist.train.labels)
    test_accuracy = get_batch_accuracy_loss(mnist.test.images, mnist.test.labels)

    print('Loss: ', total_loss / epoch_batches)
    print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)

sess.close()