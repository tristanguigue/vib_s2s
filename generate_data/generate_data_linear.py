import tensorflow as tf
import numpy as np
import time
import math
import matplotlib.pyplot as plt

seq_size = 30
hidden_size = 128
bottleneck_size = 32
encoder_output = 2 * bottleneck_size
output_size = 1
learning_rate = 0.0005
layers = 1
train_seq_size = 30
standard_mu = tf.zeros(bottleneck_size)
standard_sigma = tf.ones(bottleneck_size)
multivariate_std = tf.contrib.distributions.MultivariateNormalDiag(standard_mu, standard_sigma)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sample_bernouilli(input_values):
    return np.random.binomial(1, p=input_values)


def sample_bernouilli_tf(input_values, batch_size):
    return tf.where(
        tf.random_uniform([batch_size]) - input_values < 0,
        tf.ones([batch_size]),
        tf.zeros([batch_size]))

# samples = []
# for i in range(10000):
#     sequence = []
#     for t in range(seq_size):
#         sequence.append(t + 3 * np.sin(t) - (t / 10)**3 + np.random.normal(scale=2))
#     samples.append(sequence)
# samples = np.asarray(samples)

# min_val = np.min(samples)
# max_val = np.max(samples)
# samples = (samples - min_val) / (max_val - min_val)

# np.save('generated_samples.npy', samples)
samples = np.load('data/linear_samples.npy')

stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(hidden_size) for _ in range(layers)])

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, train_seq_size], name='x-input')
    inputs = tf.expand_dims(x, 2)
    pred_x = tf.placeholder(tf.float32, [None, train_seq_size], name='pred-input')
    pred_inputs = tf.expand_dims(pred_x, 2)

    lr = tf.placeholder(tf.float32)

with tf.variable_scope('rnn'):
    outputs, state = tf.nn.dynamic_rnn(stack, inputs, dtype=tf.float32)
    encoder_weights = tf.get_variable('encoder_weights', shape=[hidden_size, encoder_output],
        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    encoder_biases = tf.Variable(tf.constant(0.0, shape=[encoder_output]), name='encoder_biases')

    decoder_weights = tf.get_variable('decoder_weights', shape=[bottleneck_size, output_size],
        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    decoder_biases = tf.Variable(tf.constant(0.0, shape=[output_size]), name='decoder_biases')

flat_outputs = tf.reshape(outputs, [-1, hidden_size])
encoder_output = tf.matmul(flat_outputs, encoder_weights) + encoder_biases

mu = encoder_output[:, :bottleneck_size]
sigma = tf.nn.softplus(encoder_output[:, bottleneck_size:])
epsilon = tf.reshape(multivariate_std.sample(), [-1, 1])
z = mu + tf.matmul(sigma, epsilon)

decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
decoder_output = tf.reshape(decoder_output, [-1, seq_size, output_size])

loss = tf.square(tf.norm(inputs[:, 1:] - decoder_output[:, :-1], axis=1))

loss_op = tf.reduce_mean(loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_op)

predicted_sequence = []
with tf.variable_scope('rnn', reuse=True):
    # Get first prediction
    pred_rnn_state = None
    predicted_sequence = []

    # Loop to predict all the next pixels
    for i in range(train_seq_size - 1):
        true_vals = tf.reshape(pred_inputs[:, i], [-1, 1, 1])

        pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
            stack, true_vals, initial_state=pred_rnn_state, dtype=tf.float32)

        pred_logits = tf.matmul(pred_outputs[:, -1], encoder_weights) + encoder_biases
        z = encoder_output[:, :bottleneck_size]
        pred_logits = tf.matmul(z, decoder_weights) + decoder_biases
        pred_pixels = pred_logits
        pred_logits = tf.squeeze(pred_logits)
        predicted_sequence.append(pred_logits)

    predicted_sequence = tf.stack(predicted_sequence)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
former_loss = None
last_update = None
train_losses = []
test_losses = []
lr_updates = 0

for epoch in range(4000):
    print('\nEpoch:', epoch)
    start = time.time()

    # batch_xs = samples[:, :15]
    # batch_ys = samples[:, 15:]
    batch_xs = samples[:500, :]
    batch_ys = samples[500:, :]

    train_loss = sess.run(loss_op, feed_dict={x: batch_xs})
    test_loss = sess.run(loss_op, feed_dict={x: batch_ys})

    _, current_loss = sess.run([train_step, loss_op], feed_dict={
        x: batch_xs,
        lr: learning_rate
    })

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print('Time: ', time.time() - start)
    print('Loss: ', current_loss)
    print('Learning rate: ', learning_rate)
    print('Train loss: ', train_loss, ', test loss: ', test_loss)

    # if not epoch % 300:
    #     pred_seq = sess.run(predicted_sequence, feed_dict={pred_x: batch_xs[0:1]})
    #     plt.plot(batch_xs[0, 1:])
    #     plt.plot(pred_seq)
    #     plt.show()

    #     pred_seq = sess.run(predicted_sequence, feed_dict={pred_x: batch_xs[1:2]})
    #     plt.plot(batch_xs[1, 1:])
    #     plt.plot(pred_seq)
    #     plt.show()

    #     pred_seq = sess.run(predicted_sequence, feed_dict={pred_x: batch_ys[0:1]})
    #     plt.plot(batch_ys[0, 1:])
    #     plt.plot(pred_seq)
    #     plt.show()

plt.plot(train_losses)
plt.plot(test_losses)
plt.show()
