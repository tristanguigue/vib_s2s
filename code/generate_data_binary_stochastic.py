import tensorflow as tf
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from tools import kl_divergence 

seq_size = 30
hidden_size = 128
output_size = 1
learning_rate = 0.0005
layers = 1
train_seq_size = 30
bottleneck_size = 128
encoder_output = 2 * bottleneck_size
standard_mu = tf.zeros(bottleneck_size)
standard_sigma = tf.ones(bottleneck_size)
multivariate_std = tf.contrib.distributions.MultivariateNormalDiag(standard_mu, standard_sigma)
beta = 5 * 10**-3


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sample_bernouilli(input_values):
    return np.random.binomial(1, p=input_values)


def sample_bernouilli_tf(input_values, batch_size):
    return tf.where(
        tf.random_uniform([batch_size]) - input_values < 0,
        tf.ones([batch_size]),
        tf.zeros([batch_size]))

samples = []
for i in range(1000):
    ps = [0.5] * 5
    sequence = sample_bernouilli(ps)
    for t in range(5, seq_size):
        si = np.sign(sum(sequence[t - 5:t]) - 5 / 2)
        if si == -1:
            p = 0.9
        else:
            p = 0.1
        s = sample_bernouilli(p)
        sequence = np.append(sequence, s)
    samples.append(sequence)
samples = np.asarray(samples)


# samples = []
# for i in range(1000):
#     sequence = []
#     for t in range(seq_size):
#         # sequence.append(sigmoid(t - (t / 5) ** 3))
#         # sequence.append(sigmoid(-100 + 10.5 * t))
#     sample = sample_bernouilli(sequence)
#     samples.append(sample)
# samples = np.asarray(samples)

print(samples)

# np.save('generated_samples.npy', samples)
samples = np.load('data/generated_samples.npy')

stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(hidden_size) for _ in range(layers)])

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, train_seq_size], name='x-input')
    inputs = tf.expand_dims(x, 2)
    pred_x = tf.placeholder(tf.float32, [None, train_seq_size], name='pred-input')
    pred_inputs = tf.expand_dims(pred_x, 2)

    lr = tf.placeholder(tf.float32)

with tf.name_scope('prior'):
    sigma0 = tf.Variable(tf.ones(bottleneck_size), name='prior-variance')
    mu0 = tf.Variable(tf.zeros(bottleneck_size), name='prior-mean')

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

true_pixels = inputs[:, 1:]
logits = decoder_output[:, :-1]
predicted_pixels = tf.round(tf.sigmoid(logits))
accurate_predictions = tf.equal(predicted_pixels, true_pixels)
accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_pixels, logits=logits)
kl_loss = kl_divergence(mu, sigma, mu0, sigma0)
kl_loss = tf.reshape(kl_loss, [-1, seq_size, output_size])[:, :-1]

if beta:
    loss_op = tf.reduce_mean(loss + beta * kl_loss)
else:
    loss_op = tf.reduce_mean(loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_op)


# with tf.variable_scope('rnn', reuse=True):
#     # Get first prediction
#     predicted_sequence = []
#     pred_rnn_state = None

#     # Loop to predict all the next pixels
#     for i in range(seq_size - 1):
#         true_vals = tf.reshape(pred_inputs[:, i], [-1, 1, 1])

#         pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
#             stack, true_vals, initial_state=pred_rnn_state, dtype=tf.float32)
#         pred_logits = tf.matmul(pred_outputs[:, -1], decoder_weights) + decoder_biases
#         pred_pixels = tf.round(tf.sigmoid(pred_logits))
#         predicted_sequence.append(tf.squeeze(pred_pixels))

#     predicted_sequence = tf.stack(predicted_sequence)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
former_loss = None
last_update = None
train_losses = []
test_losses = []
lr_updates = 0
min_test_loss = None

for epoch in range(3000):
    print('\nEpoch:', epoch)
    start = time.time()

    # batch_xs = samples[:, :15]
    # batch_ys = samples[:, 15:]
    batch_train = samples[:250, :]
    batch_test = samples[250:, :]

    train_accuracy, train_loss, s, kl, ll = sess.run([accuracy, loss_op, sigma, kl_loss, loss], feed_dict={x: batch_train})
    print(np.mean(s), np.mean(kl), np.mean(ll))
    test_accuracy, test_loss = sess.run([accuracy, loss_op], feed_dict={x: batch_test})

    _, current_loss = sess.run([train_step, loss_op], feed_dict={
        x: batch_train,
        lr: learning_rate
    })

    # if former_loss is not None and current_loss >= former_loss:
    #     if last_update is None or epoch - last_update > 3:
    #         learning_rate /= 1.5
    #         last_update = epoch
    #         lr_updates += 1
    # elif last_update is not None and epoch - last_update > 20 * (1 + 3 * epoch / 2000):
    #     learning_rate *= 1.5
    #     last_update = epoch
    #     lr_updates = 0
    # else:
    #     lr_updates = 0
    # former_loss = current_loss

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print('Time: ', time.time() - start)
    print('Loss: ', current_loss)
    print('Learning rate: ', learning_rate)
    print('Train accuracy: ', train_accuracy, ', test accuracy: ', test_accuracy)
    print('Train loss: ', train_loss, ', test loss: ', test_loss)

    # if not epoch % 300:
    #     pred_seq = sess.run(predicted_sequence, feed_dict={pred_x: batch_train[0:1]})
    #     print(np.array(pred_seq).astype(int))
    #     print(np.array(batch_train[0, 1:]))

    #     pred_seq = sess.run(predicted_sequence, feed_dict={pred_x: batch_train[1:2]})
    #     print('-')
    #     print(np.array(pred_seq).astype(int))
    #     print(np.array(batch_train[1, 1:]))

    #     pred_seq = sess.run(predicted_sequence, feed_dict={pred_x: batch_test[0:1]})
    #     print('---')
    #     print(np.array(pred_seq).astype(int))
    #     print(np.array(batch_test[0, 1:]))

    if lr_updates > 5:
        break

print(min(test_losses))
plt.plot(train_losses)
plt.plot(test_losses)
plt.show()
