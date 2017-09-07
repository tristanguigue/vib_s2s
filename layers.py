import tensorflow as tf
import tensorflow.contrib.layers as layers

DROPOUT_PROB = 0.95


def stochastic_layer(x, bottleneck_size, nb_samples):
    with tf.name_scope('stochastic_layer'):
        standard_mu = tf.zeros(bottleneck_size)
        standard_sigma = tf.ones(bottleneck_size)
        multivariate_std = tf.contrib.distributions.MultivariateNormalDiag(standard_mu, standard_sigma)

    batch_size = tf.shape(x)[0]

    mu = x[:, :bottleneck_size]
    sigma = tf.log(1 + tf.exp(x[:, bottleneck_size:] - 5.0))

    epsilon = multivariate_std.sample(sample_shape=(batch_size, nb_samples))
    epsilon = tf.reduce_mean(epsilon, 1)

    z = mu + tf.multiply(epsilon, sigma)
    return z, mu, sigma


def deterministic_layer(x, bottleneck_size):
    return x[:, :bottleneck_size]


def accuracy_layer(x, y_true):
    accurate_predictions = tf.equal(x, y_true)
    return 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))


def gru_cell_wrapper(hidden_size, input_size, dropout, nb_layers):
    cell = tf.contrib.rnn.GRUCell(hidden_size)
    if dropout:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell,
            input_keep_prob=DROPOUT_PROB,
            output_keep_prob=DROPOUT_PROB,
            state_keep_prob=DROPOUT_PROB,
            input_size=input_size,
            dtype=tf.float32,
            variational_recurrent=True)

    stack = tf.contrib.rnn.MultiRNNCell([cell for _ in range(nb_layers)])
    return stack


def cnn_layer(x, img_size, seq_size, channels):
    flat_size = int(img_size / 4) ** 2

    x_image = tf.reshape(x, [-1, img_size, img_size, 1])
    conv1 = layers.conv2d(x_image, channels, [3, 3], scope='conv1', activation_fn=tf.nn.relu)
    pool1 = layers.max_pool2d(conv1, [2, 2], scope='pool1')
    conv2 = layers.conv2d(pool1, channels, [3, 3], scope='conv2', activation_fn=tf.nn.relu)
    pool2 = layers.max_pool2d(conv2, [2, 2], scope='pool2')
    return tf.reshape(pool2, [-1, seq_size, flat_size * channels])
