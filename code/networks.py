import tensorflow as tf
from tools import tf_binarize
from abc import ABC


class StochasticNetwork(ABC):
    def __init__(self, bottleneck_size, update_prior):
        self.bottleneck_size = bottleneck_size
        self.update_prior = update_prior
        with tf.name_scope('stochastic_layer'):
            standard_mu = tf.zeros(bottleneck_size)
            standard_sigma = tf.ones(bottleneck_size)
            self.multivariate_std = tf.contrib.distributions.MultivariateNormalDiag(standard_mu, standard_sigma)

        with tf.name_scope('prior'):
            self.sigma0 = tf.Variable(tf.ones(bottleneck_size), name='prior-variance')
            self.mu0 = tf.Variable(tf.zeros(bottleneck_size), name='prior-mean')

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))

    def bias_variable(self, name, shape):
        return tf.Variable(tf.constant(0.0, shape=shape), name=name)


class StochasticFeedForwardNetwork(StochasticNetwork):
    def __init__(self, input_size, hidden_size, bottleneck_size, output_size, update_prior):
        super().__init__(bottleneck_size, update_prior)
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
    def __init__(self, seq_size, hidden_size, bottleneck_size, output_size, layers, update_prior,
                 lstm=True, binary=True):
        super().__init__(bottleneck_size, update_prior)
        self.seq_size = seq_size
        self.output_size = output_size

        if lstm:
            cell = tf.contrib.rnn.GRUCell(hidden_size)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(hidden_size)

        stack = tf.contrib.rnn.MultiRNNCell(
            [cell for _ in range(layers)])
        encoder_output = 2 * bottleneck_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            if binary:
                self.inputs = tf_binarize(self.x)
            else:
                self.inputs = self.x
            self.inputs = tf.expand_dims(self.inputs, 2)

        with tf.name_scope('encoder'):
            out_weights = self.weight_variable('out_weights', [hidden_size, encoder_output])
            out_biases = self.bias_variable('out_biases', [encoder_output])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        with tf.variable_scope('rnn'):
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

        with tf.variable_scope('rnn', reuse=True):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(stack, self.inputs, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size])
            encoder_pred_output = tf.matmul(flat_pred_outputs, out_weights) + out_biases
            mu = encoder_pred_output[:, :bottleneck_size]
            decoder_pred_output = tf.matmul(mu, decoder_weights) + decoder_biases
            decoder_pred_output = tf.reshape(decoder_pred_output, [-1, seq_size, output_size])
            self.predicted_sequence = tf.squeeze(tf.cast(
                tf.round(tf.sigmoid(self.decoder_output)), tf.int32))


class Seq2Seq(StochasticNetwork):
    def __init__(self, seq_size, partial_seq_size, output_seq_size, hidden_size,
                 bottleneck_size, output_size, layers, update_prior):
        super().__init__(bottleneck_size, update_prior)
        self.partial_seq_size = partial_seq_size
        self.output_seq_size = output_seq_size
        self.output_size = output_size

        stack = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(layers)])
        encoder_output = 2 * bottleneck_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            self.inputs = tf.expand_dims(tf_binarize(self.x), 2)
            pred_inputs = self.inputs[:, self.partial_seq_size:
                                      self.partial_seq_size + self.output_seq_size]

        with tf.name_scope('encoder'):
            out_weights = self.weight_variable('out_weights', [hidden_size, encoder_output])
            out_biases = self.bias_variable('out_biases', [encoder_output])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable(
                'decoder_weights', [bottleneck_size, self.output_size + 2 * hidden_size * layers])
            decoder_biases = self.bias_variable(
                'decoder_biases', [self.output_size + 2 * hidden_size * layers])

        with tf.name_scope('rnn_output'):
            rnn_out_weights = self.weight_variable('rnn_out_weights', [hidden_size, output_size])
            rnn_out_biases = self.bias_variable('rnn_out_biases', [output_size])

        with tf.variable_scope('rnn'):
            outputs, state = tf.nn.dynamic_rnn(
                stack, self.inputs[:, :self.partial_seq_size], dtype=tf.float32)

        encoder_output = tf.matmul(outputs[:, -1], out_weights) + out_biases

        self.mu = encoder_output[:, :bottleneck_size]
        self.sigma = tf.nn.softplus(encoder_output[:, bottleneck_size:])
        epsilon = tf.reshape(self.multivariate_std.sample(), [-1, 1])

        z = self.mu + tf.matmul(self.sigma, epsilon)

        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        pred_logits = decoder_output[:, :output_size]

        new_state = decoder_output[:, output_size:]
        new_state = tf.reshape(new_state, [layers, 2, -1, hidden_size])
        new_state = tf.unstack(new_state)
        new_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(new_state[l][0], new_state[l][1])
                for l in range(layers)])

        with tf.variable_scope('rnn', reuse=True):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(stack, pred_inputs, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size])
            seq_logits = tf.matmul(flat_pred_outputs, rnn_out_weights) + rnn_out_biases
            seq_logits = tf.reshape(seq_logits, [-1, output_seq_size])
            seq_logits = tf.concat([pred_logits, seq_logits[:, :-1]], 1)

            self.predicted_sequence = tf.cast(tf.round(tf.sigmoid(seq_logits)), tf.int32)

            self.pred_x_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=seq_logits, labels=tf.squeeze(pred_inputs)))

        # self.predicted_sequence = tf.transpose(tf.stack(predicted_sequence))
        accurate_predictions = tf.equal(self.predicted_sequence,
                                        tf.cast(tf.squeeze(pred_inputs), tf.int32))
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))

        sampled_sequence = []
        with tf.variable_scope('rnn', reuse=True):
            # Get first cross entropies
            pred_rnn_state = new_state
            pred_pixels = tf.round(tf.sigmoid(pred_logits))

            pred_pixels = tf.reshape(pred_pixels, [-1, 1, 1])
            pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
                stack, tf.cast(pred_pixels, tf.float32),
                initial_state=pred_rnn_state, dtype=tf.float32)

            # Loop to predict all the next pixels
            for i in range(output_seq_size - 1):
                pred_logits = tf.matmul(pred_outputs[:, -1], rnn_out_weights) + rnn_out_biases
                pred_pixels = tf.round(tf.sigmoid(pred_logits))
                sampled_sequence.append(tf.cast(tf.squeeze(pred_pixels), tf.int32))

                pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
                    stack, tf.reshape(pred_pixels, [-1, 1, 1]), initial_state=pred_rnn_state,
                    dtype=tf.float32)

        self.sampled_sequence = tf.stack(sampled_sequence)


class StochasticCharRNN(StochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, output_size, layers, update_prior):
        super().__init__(bottleneck_size, update_prior)
        self.seq_size = seq_size
        self.output_size = output_size

        stack = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(layers)])
        encoder_output = 2 * bottleneck_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.int64, [None, seq_size], name='x-input')
            self.embedding = tf.get_variable('embedding', [output_size, hidden_size])

        with tf.name_scope('encoder'):
            out_weights = self.weight_variable('out_weights', [hidden_size, encoder_output])
            out_biases = self.bias_variable('out_biases', [encoder_output])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        self.inputs = tf.nn.embedding_lookup(self.embedding, self.x)
        outputs, state = tf.nn.dynamic_rnn(stack, self.inputs, dtype=tf.float32)
        flat_outputs = tf.reshape(outputs, [-1, hidden_size])

        encoder_output = tf.matmul(flat_outputs, out_weights) + out_biases

        self.mu = encoder_output[:, :bottleneck_size]
        self.sigma = tf.nn.softplus(encoder_output[:, bottleneck_size:])
        epsilon = tf.reshape(self.multivariate_std.sample(), [-1, 1])
        z = self.mu + tf.matmul(self.sigma, epsilon)

        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        self.decoder_output = tf.reshape(decoder_output, [-1, seq_size, output_size])

        true_char = self.x[:, 1:]
        predicted_char = tf.arg_max(self.decoder_output[:, :-1], 2)
        accurate_predictions = tf.equal(predicted_char, true_char)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))
