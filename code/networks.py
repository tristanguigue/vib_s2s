import tensorflow as tf
from tools import tf_binarize
from abc import ABC


class StochasticNetwork(ABC):
    def __init__(self, bottleneck_size):
        self.bottleneck_size = bottleneck_size
        with tf.name_scope('stochastic_layer'):
            standard_mu = tf.zeros(bottleneck_size)
            standard_sigma = tf.ones(bottleneck_size)
            self.multivariate_std = tf.contrib.distributions.MultivariateNormalDiag(standard_mu, standard_sigma)

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))

    def bias_variable(self, name, shape):
        return tf.Variable(tf.constant(0.0, shape=shape), name=name)


class StochasticFeedForwardNetwork(StochasticNetwork):
    def __init__(self, input_size, hidden_size, bottleneck_size, output_size):
        super().__init__(bottleneck_size)
        encoder_output = 2 * bottleneck_size

        # Variables
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, output_size], name='y-input')

        with tf.name_scope('encoder_h1'):
            self.h1_weights = self.weight_variable('h1_weights', [input_size, hidden_size])
            self.h1_biases = self.bias_variable('h1_biases', [hidden_size])

        with tf.name_scope('encoder_h2'):
            self.h2_weights = self.weight_variable('h2_weights', [hidden_size, hidden_size])
            self.h2_biases = self.bias_variable('h2_biases', [hidden_size])

        with tf.name_scope('encoder_out'):
            self.enc_out_weights = self.weight_variable('enc_out_weights', [hidden_size, encoder_output])
            self.enc_out_biases = self.bias_variable('enc_out_biases', [encoder_output])

        with tf.name_scope('decoder'):
            self.decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            self.decoder_biases = self.bias_variable('decoder_biases', [output_size])

        # Model
        hidden1 = tf.nn.relu(tf.matmul(self.x, self.h1_weights) + self.h1_biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, self.h2_weights) + self.h2_biases)
        encoder_output = tf.matmul(hidden2, self.enc_out_weights) + self.enc_out_biases

        self.mu = encoder_output[:, :bottleneck_size]
        self.sigma = tf.nn.softplus(encoder_output[:, bottleneck_size:])

        epsilon = tf.reshape(self.multivariate_std.sample(), [-1, 1])
        z = self.mu + tf.matmul(self.sigma, epsilon)

        self.decoder_output = tf.matmul(z, self.decoder_weights) + self.decoder_biases

        accurate_predictions = tf.equal(tf.arg_max(self.decoder_output, 1), tf.argmax(self.y_true, 1))
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))


class StochasticRNN(StochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, output_size, layers):
        super().__init__(bottleneck_size)
        self.seq_size = seq_size
        self.output_size = output_size

        stack = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(layers)])
        encoder_output = 2 * bottleneck_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            self.inputs = tf.expand_dims(tf_binarize(self.x), 2)

        with tf.name_scope('encoder'):
            out_weights = self.weight_variable('out_weights', [hidden_size, encoder_output])
            out_biases = self.bias_variable('out_biases', [encoder_output])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        outputs, state = tf.nn.dynamic_rnn(stack, self.inputs, dtype=tf.float32)
        flat_outputs = tf.reshape(outputs, [-1, hidden_size])

        encoder_output = tf.matmul(flat_outputs, out_weights) + out_biases

        self.mu = encoder_output[:, :bottleneck_size]
        self.sigma = tf.nn.softplus(encoder_output[:, bottleneck_size:])
        epsilon = tf.reshape(self.multivariate_std.sample(), [-1, 1])
        z = self.mu + tf.matmul(self.sigma, epsilon)

        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        self.decoder_output = tf.reshape(decoder_output, [-1, seq_size, output_size])

        true_pixels = self.inputs[:, 1:]
        predicted_pixels = tf.round(tf.sigmoid(self.decoder_output[:, :-1]))
        accurate_predictions = tf.equal(predicted_pixels, true_pixels)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))
