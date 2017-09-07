import tensorflow as tf
from tools import tf_binarize
from abc import ABC
import numpy as np
import tensorflow.contrib.layers as layers
from layers import stochastic_layer, deterministic_layer, accuracy_layer
from layers import gru_cell_wrapper, cnn_layer


class StochasticNetwork(ABC):
    """Abstract Stochastic Network with Information Bottleneck. Initialise
    the stochastic layer and the marginal parameters if necessary.

    Attributes:
        bottleneck_size: The size of the bottleneck layer
        update_marginal: Whether to parametrise the marginal on the representation layer
    """

    def __init__(self, hidden_size, bottleneck_size, update_marginal, nb_samples, dropout):
        self.update_marginal = update_marginal
        self.bottleneck_size = bottleneck_size
        self.nb_samples = nb_samples
        self.dropout = dropout

        with tf.name_scope('marginal'):
            self.sigma0 = tf.Variable(tf.ones(bottleneck_size), name='marginal-variance')
            self.mu0 = tf.Variable(tf.zeros(bottleneck_size), name='marginal-mean')

        with tf.name_scope('encoder_out'):
            self.out_weights = self.weight_variable('out_weights', [hidden_size, 2 * bottleneck_size])
            self.out_biases = self.bias_variable('out_biases', [2 * bottleneck_size])

    def weight_variable(self, name, shape):
        """Create weight variables of linear transformations with the
        xavier initialisation.

        Args:
            name: name of the variable
            shape: shape of the variable

        Return:
            The created variable
        """
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(
            uniform=False))

    def bias_variable(self, name, shape):
        """Create bias variables of linear transformations initialised at 0.

        Args:
            name: name of the variable
            shape: shape of the variable

        Return:
            The created variable
        """
        return tf.Variable(tf.constant(0.0, shape=shape), name=name)

    def representation_layer(self, x):
        """Gets the representation layer z using a stochastic or
        deterministic encoding.

        Args:
            x: the raw output of the decoder

        Return:
            The representation layer z
        """
        encoder_out = tf.matmul(x, self.out_weights) + self.out_biases
        if self.dropout:
            return deterministic_layer(encoder_out, self.bottleneck_size)
        z, mu, sigma = stochastic_layer(encoder_out, self.bottleneck_size, self.nb_samples)
        self.mu = mu
        self.sigma = sigma
        return z


class S2SStochasticNetwork(StochasticNetwork):
    """Abstract Sequence to Sequence stochastic network.

    Attributes:
        first_stack: The stack of the encoder RNN
        second_stack: The stack of the decoder RNN
        hidden_size1: The hidden units of the encoder RNN
        hidden_size2: The hidden units of the decoder RNN
        nb_layers: The number of layers of both RNNs
        output_seq_size: The number of items to predict
        binary: Whether the data is binary or not
    """

    def __init__(self, hidden_size1, hidden_size2, bottleneck_size, update_marginal, dropout,
                 nb_layers, input_size, output_size, output_seq_size, binary):
        super().__init__(hidden_size1, bottleneck_size, update_marginal, nb_samples, dropout)

        self.first_stack = gru_cell_wrapper(hidden_size1, input_size, dropout, nb_layers)
        self.second_stack = gru_cell_wrapper(hidden_size2, output_size, dropout, nb_layers)
        self.hidden_size1 = hidden_size2
        self.hidden_size2 = hidden_size2
        self.nb_layers = nb_layers
        self.output_seq_size = output_seq_size
        self.binary = binary

        with tf.name_scope('rnn_output'):
            self.rnn_out_weights = self.weight_variable(
                'rnn_out_weights', [hidden_size2, output_size])
            self.rnn_out_biases = self.bias_variable('rnn_out_biases', [output_size])

        with tf.name_scope('decoder'):
            self.dec_weights_first_input = self.weight_variable(
                'dec_weights_first_input', [bottleneck_size, output_size])
            self.dec_biases_first_input = self.bias_variable(
                'dec_biases_first_input', [output_size])
            self.dec_weights_state = self.weight_variable(
                'dec_weights_state', [bottleneck_size, hidden_size2 * nb_layers])
            self.dec_biases_state = self.bias_variable(
                'dec_biases_state', [hidden_size2 * nb_layers])

    def encoder_layer(self, x):
        """The encoder of a s2s network

        Args:
            x: the input values

        Return:
            The representation layer z
        """
        with tf.variable_scope('encoder_rnn'):
            outputs, state = tf.nn.dynamic_rnn(self.first_stack, x, dtype=tf.float32)
        return self.representation_layer(outputs[:, -1])

    def decoder_layer(self, z, y_true):
        """The decoder of a s2s network

        Args:
            z: the stochastic encoding
            y_true: the true value to feed the decoder

        Return:
            The logits output of the decoder
        """
        first_logits = tf.matmul(z, self.dec_weights_first_input) + self.dec_biases_first_input
        new_state = tf.matmul(z, self.dec_weights_state) + self.dec_biases_state
        new_state = tf.reshape(new_state, [self.nb_layers, -1, self.hidden_size2])
        self.new_state = tuple(tf.unstack(new_state))

        with tf.variable_scope('pred_rnn'):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(
                self.second_stack, y_true, initial_state=new_state, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, self.hidden_size2])
            seq_logits = tf.matmul(flat_pred_outputs, self.rnn_out_weights) + self.rnn_out_biases
            seq_logits = tf.reshape(seq_logits, [-1, self.output_seq_size])
            return first_logits, seq_logits

    def sample_sequence(self, first_logits, pred_rnn_state):
        sampled_sequence = []
        sample_logits = []
        with tf.variable_scope('sampled_rnn'):
            if self.binary:
                pred_inputs = tf.round(tf.sigmoid(first_logits))
            else:
                pred_inputs = first_logits
            sampled_sequence.append(tf.squeeze(pred_inputs))
            sample_logits.append(first_logits)
            pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
                self.second_stack, tf.cast(tf.reshape(pred_inputs, [-1, 1, 1]), tf.float32),
                initial_state=pred_rnn_state, dtype=tf.float32)

        with tf.variable_scope('sampled_rnn', reuse=True):
            # Loop to predict all the next pixels
            for i in range(self.output_seq_size - 1):
                pred_logits = tf.matmul(pred_outputs[:, -1], self.rnn_out_weights) + self.rnn_out_biases
                if self.binary:
                    pred_inputs = tf.round(tf.sigmoid(pred_logits))
                else:
                    pred_inputs = pred_logits
                sampled_sequence.append(tf.squeeze(pred_inputs))
                sample_logits.append(pred_logits)
                pred_outputs, pred_rnn_state = tf.nn.dynamic_rnn(
                    self.second_stack, tf.reshape(pred_inputs, [-1, 1, 1]), initial_state=pred_rnn_state,
                    dtype=tf.float32)
        self.sampled_sequence = tf.transpose(tf.stack(sampled_sequence))


class S2LStochasticNetwork(StochasticNetwork):
    """Abstract Sequence to Label stochastic network.

    Attributes:
        stack: The stack of the RNN
        dec_weights: The weights of the decoder
        dec_biases: The bias of the decoder
    """

    def __init__(self, hidden_size, bottleneck_size, update_marginal, dropout,
                 nb_layers, input_size, output_size, nb_samples):
        super().__init__(hidden_size, bottleneck_size, update_marginal, nb_samples, dropout)

        self.stack = gru_cell_wrapper(hidden_size, input_size, dropout, nb_layers)

        with tf.name_scope('decoder'):
            self.dec_weights = self.weight_variable('dec_weights', [bottleneck_size, output_size])
            self.dec_biases = self.bias_variable('dec_biases', [output_size])

    def encoder_layer(self, x):
        """The encoder of a sequence to label network

        Args:
            x: the input values

        Return:
            The representation layer z
        """
        with tf.variable_scope('encoder_rnn'):
            outputs, state = tf.nn.dynamic_rnn(self.stack, x, dtype=tf.float32)
        return self.representation_layer(outputs[:, -1])


class StochasticFeedForwardNetwork(StochasticNetwork):
    """Simple feedforward network with two hidden layers
    the stochastic layer and the marginal parameters if necessary.

    Attributes:
        x: The input
        y_true: The expected output
        output: the predicted logits
        accuracy: the percentage of accurate predictions
    """

    def __init__(self, input_size, hidden_size, bottleneck_size, output_size, update_marginal,
                 nb_samples, dropout):
        super().__init__(hidden_size, bottleneck_size, update_marginal, nb_samples, dropout)

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

        with tf.name_scope('decoder'):
            decoder_weights = self.weight_variable('decoder_weights', [bottleneck_size, output_size])
            decoder_biases = self.bias_variable('decoder_biases', [output_size])

        # Model
        hidden1 = tf.nn.relu(tf.matmul(self.x, h1_weights) + h1_biases)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, h2_weights) + h2_biases)
        print(hidden2.get_shape())
        z = self.representation_layer(hidden2)
        print(z.get_shape())
        self.output = tf.matmul(z, decoder_weights) + decoder_biases
        self.accuracy = accuracy_layer(tf.arg_max(self.output, 1), tf.argmax(self.y_true, 1))


class Seq2Seq(S2SStochasticNetwork):
    """A Sequence to sequence prediction model with stochastic layer.

    Attributes:
        x: The input
        y_true: The expected output
        mu: The mean of the representation layer Z
        sigma: the diagonal covariance of the representation layer Z
        output: the predicted logits
        accuracy: the percentage of accurate predictions
    """

    def __init__(self, partial_seq_size, output_seq_size, hidden_size1, hidden_size2,
                 bottleneck_size, output_size, nb_layers, nb_samples, update_marginal,
                 dropout, binary=True):
        super().__init__(hidden_size1, hidden_size2, bottleneck_size, update_marginal, dropout,
                         nb_layers, 1, output_size, output_seq_size, binary)
        seq_size = partial_seq_size + output_seq_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            if binary:
                self.inputs = tf_binarize(self.x)
            else:
                self.inputs = self.x
            self.inputs = tf.expand_dims(self.inputs, 2)
            true_seq = self.inputs[:, partial_seq_size: partial_seq_size + output_seq_size]
            self.y_true = tf.squeeze(true_seq)

        z = self.encoder_layer(self.inputs[:, :partial_seq_size])
        first_logits, seq_logits = self.decoder_layer(z, true_seq)
        self.output = tf.concat([first_logits, seq_logits[:, :-1]], 1)
        self.predicted_sequence = tf.cast(tf.round(tf.sigmoid(self.output)), tf.int32)
        self.accuracy = accuracy_layer(self.predicted_sequence, tf.cast(self.y_true, tf.int32))


class Seq2SeqCont(S2SStochasticNetwork):
    """A Sequence to sequence prediction model for continuous data.

    Attributes:
        x: The input
        y_true: The expected output
        output: the predicted values
        output_sigma: the variance of the decoding approximation
        accuracy: the percentage of accurate predictions
    """

    def __init__(self, partial_seq_size, output_seq_size, hidden_size1, hidden_size2,
                 bottleneck_size, output_size, nb_layers, nb_samples, update_marginal,
                 dropout):
        super().__init__(hidden_size1, hidden_size2, bottleneck_size, update_marginal, dropout,
                         nb_layers, 1, output_size, output_seq_size, binary)
        seq_size = partial_seq_size + output_seq_size

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            self.inputs = tf.expand_dims(self.x, 2)
            true_seq = self.inputs[:, partial_seq_size: partial_seq_size + output_seq_size]
            self.y_true = tf.squeeze(true_seq)

        with tf.name_scope('decoder'):
            dec_weights_first_mu = self.weight_variable(
                'dec_weights_first_mu', [bottleneck_size, output_size])
            dec_biases_first_mu = self.bias_variable(
                'dec_biases_first_mu', [self.output_size])
            dec_weights_first_sigma = self.weight_variable(
                'dec_weights_first_sigma', [bottleneck_size, output_size])
            dec_biases_first_sigma = self.bias_variable(
                'dec_biases_first_sigma', [self.output_size])
            dec_weights_state = self.weight_variable(
                'dec_weights_state', [bottleneck_size, hidden_size2 * nb_layers])
            dec_biases_state = self.bias_variable(
                'dec_biases_state', [hidden_size2 * nb_layers])

        with tf.name_scope('rnn_output'):
            out_mu_weights = self.weight_variable('out_mu_weights', [hidden_size2, output_size])
            out_mu_biases = self.bias_variable('out_mu_biases', [output_size])
            out_sigma_weights = self.weight_variable('out_sigma_weights', [hidden_size2, output_size])
            out_sigma_biases = self.bias_variable('out_sigma_biases', [output_size])

        z = self.encoder_layer(self.inputs[:, :partial_seq_size])

        first_mu = tf.nn.sigmoid(tf.matmul(z, dec_weights_first_mu) + dec_biases_first_mu)
        first_sigma = tf.nn.softplus(tf.matmul(z, dec_weights_first_sigma) + dec_biases_first_sigma)
        new_state = tf.matmul(z, dec_weights_state) + dec_biases_state
        new_state = tf.reshape(new_state, [nb_layers, -1, hidden_size2])
        new_state = tuple(tf.unstack(new_state))

        with tf.variable_scope('pred_rnn'):
            pred_outputs, pred_state = tf.nn.dynamic_rnn(
                self.second_stack, true_seq, initial_state=new_state, dtype=tf.float32)
            flat_pred_outputs = tf.reshape(pred_outputs, [-1, hidden_size2])
            mus = tf.nn.sigmoid(tf.matmul(flat_pred_outputs, out_mu_weights) + out_mu_biases)
            mus = tf.reshape(mus, [-1, output_seq_size])
            sigmas = tf.nn.softplus(tf.matmul(flat_pred_outputs, out_sigma_weights) + out_sigma_biases)
            sigmas = tf.reshape(mus, [-1, output_seq_size])
            self.output = tf.concat([first_mu, mus[:, :-1]], 1)
            self.output_sigma = tf.concat([first_sigma, sigmas[:, :-1]], 1)

        self.predicted_sequence = tf.cast(tf.round(tf.sigmoid(self.output)), tf.int32)
        self.accuracy = accuracy_layer(self.predicted_sequence, tf.cast(self.y_true, tf.int32))


class Seq2Label(S2LStochasticNetwork):
    def __init__(self, seq_size, hidden_size, bottleneck_size, input_size, output_size,
                 nb_layers, nb_samples, update_marginal, dropout):
        super().__init__(hidden_size, bottleneck_size, update_marginal, dropout,
                         nb_layers, input_size, output_size, nb_samples)

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, output_size], name='y-input')
            self.y_true_digit = tf.argmax(self.y_true, axis=1)

        z = self.encoder_layer(self.x)
        self.output = tf.matmul(z, self.dec_weights) + self.dec_biases
        self.accuracy = accuracy_layer(tf.argmax(self.output, axis=1), self.y_true_digit)


class Seq2Labels(S2SStochasticNetwork):
    def __init__(self, seq_size, hidden_size1, hidden_size2, bottleneck_size, input_size, output_size,
                 nb_layers, nb_samples, update_marginal, dropout=False):
        super().__init__(hidden_size1, hidden_size2, bottleneck_size, update_marginal, dropout,
                         nb_layers, input_size, output_size, output_seq_size, binary)

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, seq_size, output_size], name='y-input')
            self.y_true_digits = tf.argmax(self.y_true, axis=2)

        z = self.encoder_layer(self.x)
        first_logits, seq_logits = self.decoder_layer(z, self.y_true)
        self.output = tf.concat([tf.expand_dims(first_logits, 1), seq_logits[:, :-1, :]], 1)
        self.predicted_sequence = tf.argmax(self.output, axis=2)
        self.accuracy = accuracy_layer(self.predicted_sequence, self.y_true_digit)


class Seq2LabelsCNN(S2SStochasticNetwork):
    def __init__(self, seq_size, hidden_size1, hidden_size2, bottleneck_size, input_size, output_size,
                 nb_layers, nb_samples, channels, update_marginal, dropout):
        super().__init__(hidden_size1, hidden_size2, bottleneck_size, update_marginal, dropout,
                         nb_layers, input_size, output_size, output_seq_size, binary)
        img_size = int(np.sqrt(input_size))

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, output_size], name='y-input')
            y_true = tf.reshape(self.y_true, [-1, seq_size, output_size])
            self.y_true_digits = tf.argmax(y_true, axis=2)

        rnn_inputs = cnn_layer(self.x, img_size, channels, seq_size)
        z = self.encoder_layer(rnn_inputs)
        first_logits, seq_logits = self.decoder_layer(z, y_true)
        self.output = tf.concat([tf.expand_dims(first_logits, 1), seq_logits[:, :-1, :]], 1)
        self.predicted_sequence = tf.argmax(self.output, axis=2)
        self.accuracy = accuracy_layer(self.predicted_sequence, self.y_true_digit)


class Seq2LabelCNN(S2LStochasticNetwork):
    """A Sequence to label model using CNN to feed the RNN encoder.

    Attributes:
        x: The input
        y_true: The expected output
        output: the predicted values
        accuracy: the percentage of accurate predictions
    """

    def __init__(self, seq_size, hidden_size, bottleneck_size, input_size, output_size,
                 nb_layers, nb_samples, channels, update_marginal, dropout):
        super().__init__(hidden_size, bottleneck_size, update_marginal, dropout,
                         nb_layers, input_size, output_size, nb_samples)
        img_size = int(np.sqrt(input_size))

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size, input_size], name='x-input')
            self.y_true = tf.placeholder(tf.float32, [None, output_size], name='y-input')
            self.y_true_digits = tf.argmax(self.y_true, axis=1)
        rnn_inputs = cnn_layer(self.x, img_size, channels, seq_size)
        z = self.encoder_layer(rnn_inputs)
        self.output = tf.matmul(z, self.dec_weights) + self.dec_biases
        self.accuracy = accuracy_layer(tf.argmax(self.output, axis=1), self.y_true_digit)


class Seq2Pixel(S2LStochasticNetwork):
    """A Sequence to label model for pixel prediction.

    Attributes:
        x: The input
        y_true: The expected output
        output: the predicted values
        accuracy: the percentage of accurate predictions
    """
    def __init__(self, partial_seq_size, hidden_size, bottleneck_size, output_size,
                 nb_layers, nb_samples, update_marginal, dropout):
        super().__init__(hidden_size, bottleneck_size, update_marginal, dropout,
                         nb_layers, 1, output_size, nb_samples)
        seq_size = partial_seq_size + 1

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, [None, seq_size], name='x-input')
            self.inputs = tf.expand_dims(tf_binarize(self.x), 2)
            self.y_true = self.inputs[:, -1]

        z = self.encoder_layer(self.inputs[:, :partial_seq_size])
        self.output = tf.matmul(z, self.dec_weights) + self.dec_biases
        self.accuracy = accuracy_layer(tf.round(tf.sigmoid(self.output)), self.y_true)
