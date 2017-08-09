import tensorflow as tf
from tools import tf_binarize
from abc import ABC
import numpy as np
import tensorflow.contrib.layers as layers


class StochasticNetwork(ABC):
    def __init__(self, bottleneck_size, update_prior):
        self.bottleneck_size = bottleneck_size
        self.update_prior = update_prior
        self.is_training = tf.placeholder(tf.bool, name='is-training')

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
    def __init__(self, input_size, hidden_size, bottleneck_size, output_size, update_prior,
                 nb_samples):
        super().__init__(bottleneck_size, update_prior)

        # Variables
        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, output_size], name='y-input')

        with tf.name_scope('encoder_h1'):
            h1_weights = self.weight_variable('h1_weights', [input_size, hidden_size])
            h1_biases = self.bias_variable('h1_biases', [hidden_size])

        with tf.name_scope('encoder_h2'):
            h2_weights = self.weight_variable('h2_weights', [hidden_size, hidden_size])
            h2_biases = self.bias_variable('h2_biases', [hidden_size])

        with tf.name_scope('encoder_out'):
            out_weights_mu = self.weight_variable('out_weights_mu', [hidden_size, bottleneck_size])
            out_biases_mu = self.bias_variable('out_biases_mu', [bottleneck_size])
            out_weights_sigma = self.weight_variable('out_weights_logvar', [hidden_size, bottleneck_size])
            out_biases_sigma = self.bias_variable('out_biases_logvar', [bottleneck_size])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        # Model
        hidden1 = tf.nn.relu(tf.matmul(self.x, h1_weights) + h1_biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, h2_weights) + h2_biases)

        self.mu = tf.matmul(hidden2, out_weights_mu) + out_biases_mu
        self.sigma = tf.nn.softplus(tf.matmul(hidden2, out_weights_sigma) + out_biases_sigma)

        batch_size = tf.shape(self.x)[0]
        epsilon = self.multivariate_std.sample(sample_shape=(batch_size, nb_samples))
        epsilon = tf.reduce_mean(epsilon, 1)

        z = self.mu + tf.multiply(epsilon, self.sigma)
        self.output = tf.matmul(z, decoder_weights) + decoder_biases

        accurate_predictions = tf.equal(tf.arg_max(self.output, 1), tf.argmax(self.y_true, 1))
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))


class StochasticRNN(StochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, output_size, nb_layers, nb_samples,
                 update_prior=True, lstm=True, binary=True, do_batch_norm=False):
        super().__init__(bottleneck_size, update_prior)
        self.seq_size = seq_size
        self.output_size = output_size
        self.bottleneck_size = bottleneck_size

        if lstm:
            cell = tf.contrib.rnn.GRUCell(hidden_size)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
        stack = tf.contrib.rnn.MultiRNNCell([cell for _ in range(nb_layers)])

        with tf.name_scope('prior'):
            self.sigma0 = tf.Variable(tf.ones([seq_size, bottleneck_size]), name='prior-variance')
            self.mu0 = tf.Variable(tf.zeros([seq_size, bottleneck_size]), name='prior-mean')

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            if binary:
                self.inputs = tf_binarize(self.x)
            else:
                self.inputs = self.x
            self.inputs = tf.expand_dims(self.inputs, 2)
            self.y_true = self.inputs[:, 1:]

        with tf.name_scope('encoder'):
            out_weights_mu = self.weight_variable('out_weights_mu', [hidden_size, bottleneck_size])
            out_biases_mu = self.bias_variable('out_biases_mu', [bottleneck_size])
            out_weights_logvar = self.weight_variable('out_weights_logvar', [hidden_size, bottleneck_size])
            out_biases_logvar = self.bias_variable('out_biases_logvar', [bottleneck_size])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        with tf.variable_scope('rnn'):
            outputs, state = tf.nn.dynamic_rnn(stack, self.inputs, dtype=tf.float32)
            if do_batch_norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_training)

        flat_outputs = tf.reshape(outputs, [-1, hidden_size])
        self.mu = tf.matmul(flat_outputs, out_weights_mu) + out_biases_mu
        self.sigma = tf.nn.softplus(tf.matmul(flat_outputs, out_weights_logvar) + out_biases_logvar)

        batch_size = tf.shape(self.x)[0]
        epsilon = self.multivariate_std.sample(sample_shape=(batch_size * seq_size, nb_samples))
        epsilon = tf.reduce_mean(epsilon, 1)

        z = self.mu + tf.multiply(self.sigma, epsilon)
        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        self.output = tf.reshape(decoder_output, [-1, seq_size, output_size])[:, :-1]

        predicted_pixels = tf.round(tf.sigmoid(self.output))
        accurate_predictions = tf.equal(predicted_pixels, self.y_true)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))

        with tf.variable_scope('rnn', reuse=True):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(stack, self.inputs, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size])
            mu = tf.matmul(flat_pred_outputs, out_weights_mu) + out_biases_mu
            decoder_pred_output = tf.matmul(mu, decoder_weights) + decoder_biases
            decoder_pred_output = tf.reshape(decoder_pred_output, [-1, seq_size, output_size])
            if binary:
                decoder_pred_output = tf.cast(
                    tf.round(tf.sigmoid(decoder_pred_output)), tf.int32)

            self.predicted_sequence = tf.squeeze(decoder_pred_output)


class Seq2Seq(StochasticNetwork):
    def __init__(self, partial_seq_size, output_seq_size, hidden_size,
                 bottleneck_size, output_size, nb_layers, nb_samples, update_prior,
                 binary=True):
        super().__init__(bottleneck_size, update_prior)
        self.partial_seq_size = partial_seq_size
        self.output_seq_size = output_seq_size
        self.output_size = output_size
        seq_size = partial_seq_size + output_seq_size

        first_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        second_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        first_stack = tf.contrib.rnn.MultiRNNCell([first_cell for _ in range(nb_layers)])
        second_stack = tf.contrib.rnn.MultiRNNCell([second_cell for _ in range(nb_layers)])

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            if binary:
                self.inputs = tf_binarize(self.x)
            else:
                self.inputs = self.x
            self.inputs = tf.expand_dims(self.inputs, 2)
            true_seq = self.inputs[:, self.partial_seq_size:
                                   self.partial_seq_size + self.output_seq_size]
            self.y_true = tf.squeeze(true_seq)

        with tf.name_scope('encoder'):
            out_weights_mu = self.weight_variable('out_weights_mu', [hidden_size, bottleneck_size])
            out_biases_mu = self.bias_variable('out_biases_mu', [bottleneck_size])
            out_weights_sigma = self.weight_variable('out_weights_logvar', [hidden_size, bottleneck_size])
            out_biases_sigma = self.bias_variable('out_biases_logvar', [bottleneck_size])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable(
                'decoder_weights', [bottleneck_size, self.output_size + 2 * hidden_size * nb_layers])
            decoder_biases = self.bias_variable(
                'decoder_biases', [self.output_size + 2 * hidden_size * nb_layers])

        with tf.name_scope('rnn_output'):
            rnn_out_weights = self.weight_variable('rnn_out_weights', [hidden_size, output_size])
            rnn_out_biases = self.bias_variable('rnn_out_biases', [output_size])

        with tf.variable_scope('rnn'):
            outputs, state = tf.nn.dynamic_rnn(
                first_stack, self.inputs[:, :self.partial_seq_size], dtype=tf.float32)

        self.mu = tf.matmul(outputs[:, -1], out_weights_mu) + out_biases_mu
        self.sigma = tf.nn.softplus(tf.matmul(outputs[:, -1], out_weights_sigma) + out_biases_sigma)

        batch_size = tf.shape(self.x)[0]
        epsilon = self.multivariate_std.sample(sample_shape=(batch_size, nb_samples))
        epsilon = tf.reduce_mean(epsilon, 1)

        z = self.mu + tf.multiply(self.sigma, epsilon)

        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        first_logits = decoder_output[:, :output_size]

        new_state = decoder_output[:, output_size:]
        new_state = tf.reshape(new_state, [nb_layers, 2, -1, hidden_size])
        new_state = tf.unstack(new_state)
        new_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(new_state[l][0], new_state[l][1])
                for l in range(nb_layers)])

        with tf.variable_scope('pred_rnn'):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(
                second_stack, true_seq, initial_state=new_state, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size])
            seq_logits = tf.matmul(flat_pred_outputs, rnn_out_weights) + rnn_out_biases
            seq_logits = tf.reshape(seq_logits, [-1, output_seq_size])
            self.output = tf.concat([first_logits, seq_logits[:, :-1]], 1)
            self.predicted_sequence = tf.cast(tf.round(tf.sigmoid(self.output)), tf.int32)

        accurate_predictions = tf.equal(self.predicted_sequence, tf.cast(self.y_true, tf.int32))
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))

        sampled_sequence = []
        sample_logits = []
        with tf.variable_scope('sampled_rnn'):
            pred_rnn_state = new_state
            if binary:
                pred_inputs = tf.round(tf.sigmoid(first_logits))
            else:
                pred_inputs = first_logits
            sampled_sequence.append(tf.squeeze(pred_inputs))
            sample_logits.append(first_logits)
            pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
                second_stack, tf.cast(tf.reshape(pred_inputs, [-1, 1, 1]), tf.float32),
                initial_state=pred_rnn_state, dtype=tf.float32)

        with tf.variable_scope('sampled_rnn', reuse=True):
            # Loop to predict all the next pixels
            for i in range(output_seq_size - 1):
                pred_logits = tf.matmul(pred_outputs[:, -1], rnn_out_weights) + rnn_out_biases
                if binary:
                    pred_inputs = tf.round(tf.sigmoid(pred_logits))
                else:
                    pred_inputs = pred_logits
                sampled_sequence.append(tf.squeeze(pred_inputs))
                sample_logits.append(pred_logits)
                pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
                    second_stack, tf.reshape(pred_inputs, [-1, 1, 1]), initial_state=pred_rnn_state,
                    dtype=tf.float32)
        self.sampled_sequence = tf.transpose(tf.stack(sampled_sequence))


class Seq2Labels(StochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, input_size, output_size,
                 layers, nb_samples, update_prior):
        super().__init__(bottleneck_size, update_prior)
        self.seq_size = seq_size
        self.output_size = output_size

        first_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        second_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        first_stack = tf.contrib.rnn.MultiRNNCell([first_cell for _ in range(layers)])
        second_stack = tf.contrib.rnn.MultiRNNCell([second_cell for _ in range(layers)])

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, seq_size, output_size], name='y-input')
            self.y_true_digits = tf.argmax(self.y_true, axis=2)

        with tf.name_scope('encoder'):
            out_weights_mu = self.weight_variable('out_weights_mu', [hidden_size, bottleneck_size])
            out_biases_mu = self.bias_variable('out_biases_mu', [bottleneck_size])
            out_weights_sigma = self.weight_variable('out_weights_logvar', [hidden_size, bottleneck_size])
            out_biases_sigma = self.bias_variable('out_biases_logvar', [bottleneck_size])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable(
                'decoder_weights', [bottleneck_size, self.output_size + 2 * hidden_size * layers])
            decoder_biases = self.bias_variable(
                'decoder_biases', [self.output_size + 2 * hidden_size * layers])

        with tf.name_scope('rnn_output'):
            rnn_out_weights = self.weight_variable('rnn_out_weights', [hidden_size, output_size])
            rnn_out_biases = self.bias_variable('rnn_out_biases', [output_size])

        with tf.variable_scope('rnn'):
            outputs, state = tf.nn.dynamic_rnn(first_stack, self.x, dtype=tf.float32)

        self.mu = tf.matmul(outputs[:, -1], out_weights_mu) + out_biases_mu
        self.sigma = tf.nn.softplus(tf.matmul(outputs[:, -1], out_weights_sigma) + out_biases_sigma)

        batch_size = tf.shape(self.x)[0]
        epsilon = self.multivariate_std.sample(sample_shape=(batch_size, nb_samples))
        epsilon = tf.reduce_mean(epsilon, 1)

        z = self.mu + tf.multiply(self.sigma, epsilon)

        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        first_logits = decoder_output[:, :output_size]

        new_state = decoder_output[:, output_size:]
        new_state = tf.reshape(new_state, [layers, 2, -1, hidden_size])
        new_state = tf.unstack(new_state)
        new_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(new_state[l][0], new_state[l][1])
                for l in range(layers)])

        with tf.variable_scope('pred_rnn'):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(
                second_stack, self.y_true, initial_state=new_state, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size])
            seq_logits = tf.matmul(flat_pred_outputs, rnn_out_weights) + rnn_out_biases
            seq_logits = tf.reshape(seq_logits, [-1, seq_size, output_size])
            self.output = tf.concat([tf.expand_dims(first_logits, 1), seq_logits[:, :-1, :]], 1)
            self.predicted_sequence = tf.argmax(self.output, axis=2)

        accurate_predictions = tf.equal(self.predicted_sequence, self.y_true_digits)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))


class Seq2LabelsCNN(StochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, input_size, output_size,
                 nb_layers, nb_samples, channels, update_prior):
        super().__init__(bottleneck_size, update_prior)
        self.seq_size = seq_size
        self.output_size = output_size
        img_size = int(np.sqrt(input_size))
        flat_size = int(img_size / 4) ** 2

        first_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        second_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        first_stack = tf.contrib.rnn.MultiRNNCell([first_cell for _ in range(nb_layers)])
        second_stack = tf.contrib.rnn.MultiRNNCell([second_cell for _ in range(nb_layers)])

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, output_size], name='y-input')
            y_true = tf.reshape(self.y_true, [-1, seq_size, output_size])
            self.y_true_digits = tf.argmax(y_true, axis=2)

        with tf.name_scope('encoder'):
            out_weights_mu = self.weight_variable('out_weights_mu', [hidden_size, bottleneck_size])
            out_biases_mu = self.bias_variable('out_biases_mu', [bottleneck_size])
            out_weights_sigma = self.weight_variable('out_weights_logvar', [hidden_size, bottleneck_size])
            out_biases_sigma = self.bias_variable('out_biases_logvar', [bottleneck_size])

        with tf.name_scope('decoder'):
            dec_weights_first_input = self.weight_variable(
                'dec_weights_first_input', [bottleneck_size, self.output_size])
            dec_biases_first_input = self.bias_variable(
                'dec_biases_first_input', [self.output_size])
            dec_weights_state = self.weight_variable(
                'dec_weights_state', [bottleneck_size, 2 * hidden_size * nb_layers])
            dec_biases_state = self.bias_variable(
                'dec_biases_state', [2 * hidden_size * nb_layers])

        with tf.name_scope('rnn_output'):
            rnn_out_weights = self.weight_variable('rnn_out_weights', [hidden_size, output_size])
            rnn_out_biases = self.bias_variable('rnn_out_biases', [output_size])

        x_image = tf.reshape(self.x, [-1, img_size, img_size, 1])
        conv1 = layers.conv2d(x_image, channels, [3, 3], scope='conv1', activation_fn=tf.nn.relu)
        pool1 = layers.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2 = layers.conv2d(pool1, channels, [3, 3], scope='conv2', activation_fn=tf.nn.relu)
        pool2 = layers.max_pool2d(conv2, [2, 2], scope='pool2')
        inputs = tf.reshape(pool2, [-1, seq_size, flat_size * channels])

        with tf.variable_scope('rnn'):
            outputs, state = tf.nn.dynamic_rnn(first_stack, inputs, dtype=tf.float32)

        self.mu = tf.matmul(outputs[:, -1], out_weights_mu) + out_biases_mu
        self.sigma = tf.nn.softplus(tf.matmul(outputs[:, -1], out_weights_sigma) + out_biases_sigma)

        batch_size = tf.shape(inputs)[0]
        epsilon = self.multivariate_std.sample(sample_shape=(batch_size, nb_samples))
        epsilon = tf.reduce_mean(epsilon, 1)

        z = self.mu + tf.multiply(self.sigma, epsilon)

        first_logits = tf.matmul(z, dec_weights_first_input) + dec_biases_first_input
        new_state = tf.matmul(z, dec_weights_state) + dec_biases_state
        new_state = tf.reshape(new_state, [nb_layers, 2, -1, hidden_size])
        new_state = tf.unstack(new_state)
        new_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(new_state[l][0], new_state[l][1])
                for l in range(nb_layers)])

        with tf.variable_scope('pred_rnn'):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(
                second_stack, y_true, initial_state=new_state, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size])
            seq_logits = tf.matmul(flat_pred_outputs, rnn_out_weights) + rnn_out_biases
            seq_logits = tf.reshape(seq_logits, [-1, seq_size, output_size])
            self.output = tf.concat([tf.expand_dims(first_logits, 1), seq_logits[:, :-1, :]], 1)
            self.predicted_sequence = tf.argmax(self.output, axis=2)

        accurate_predictions = tf.equal(self.predicted_sequence, self.y_true_digits)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))


class Seq2Pixel(StochasticNetwork):
    def __init__(self, partial_seq_size, hidden_size, bottleneck_size, output_size,
                 nb_layers, nb_samples, update_prior, lstm):
        super().__init__(bottleneck_size, update_prior)
        self.partial_seq_size = partial_seq_size
        seq_size = partial_seq_size + 1

        if lstm:
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        else:
            cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
        stack = tf.contrib.rnn.MultiRNNCell([cell for _ in range(nb_layers)])

        encoder_output = 2 * bottleneck_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            self.inputs = tf.expand_dims(tf_binarize(self.x), 2)
            self.y_true = self.inputs[:, -1]

        with tf.name_scope('encoder'):
            out_weights = self.weight_variable('out_weights', [hidden_size, encoder_output])
            out_biases = self.bias_variable('out_biases', [encoder_output])

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        with tf.variable_scope('rnn'):
            outputs, state = tf.nn.dynamic_rnn(stack, self.inputs[:, :-1], dtype=tf.float32)

        encoder_output = tf.matmul(outputs[:, -1], out_weights) + out_biases

        self.mu = encoder_output[:, :bottleneck_size]
        self.sigma = tf.nn.softplus(encoder_output[:, bottleneck_size:])

        batch_size = tf.shape(self.x)[0]
        epsilon = self.multivariate_std.sample(sample_shape=(batch_size, nb_samples))
        epsilon = tf.reduce_mean(epsilon, 1)

        z = self.mu + tf.multiply(self.sigma, epsilon)
        self.output = tf.matmul(z, decoder_weights) + decoder_biases

        accurate_predictions = tf.equal(tf.round(tf.sigmoid(self.output)), self.y_true)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))


class StochasticCharRNN(StochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, output_size, nb_layers, update_prior):
        super().__init__(bottleneck_size, update_prior)
        self.seq_size = seq_size
        self.output_size = output_size

        stack = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(nb_layers)])
        encoder_output = 2 * bottleneck_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.int64, [None, seq_size], name='x-input')
            self.embedding = tf.get_variable('embedding', [output_size, hidden_size])
            self.y_true = self.x[:, 1:]

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
        epsilon = self.multivariate_std.sample()
        z = self.mu + tf.multiply(self.sigma, epsilon)

        decoder_output = tf.matmul(z, decoder_weights) + decoder_biases
        self.output = tf.reshape(decoder_output, [-1, seq_size, output_size])[:, :-1]

        predicted_char = tf.arg_max(self.output, 2)
        accurate_predictions = tf.equal(predicted_char, self.y_true)
        self.accuracy = 100 * tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))
