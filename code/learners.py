import tensorflow as tf
from abc import ABC, abstractmethod
from tools import kl_divergence_with_std, kl_divergence
import os
import math

DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
LOGS_PATH = 'logs/'
CHECKPOINT_PATH = 'checkpoints/'
GRADIENT_GLOBAL_NORM = 10.0


class Learner(ABC):
    """Generic learner.

    Attributes:
        net: the network to use for learning
        lr: the learning rate placeholder of the optimiser
        learning_rate: the learning rate value
        loss_op: the loss to be defined by the learners
    """

    def __init__(self, network, learning_rate, train_batch, run_name):
        self.lr = tf.placeholder(tf.float32)
        self.net = network
        self.learning_rate = learning_rate
        self.loss_op = self.loss()

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            optimizer = tf.contrib.opt.MovingAverageOptimizer(optimizer, average_decay=0.999)
            self.train_step = optimizer.minimize(self.loss_op)

        self.acc_summary = tf.summary.scalar('accuracy_summary', self.net.accuracy)
        self.train_loss_summary = tf.summary.scalar('train_loss_summary', self.loss_op)
        self.test_loss_summary = tf.summary.scalar('test_loss_summary', self.loss_op)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.test_sess = tf.Session(config=config)
        self.training_saver = optimizer.swapping_saver()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(DIR + LOGS_PATH + run_name, graph=tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())
        self.test_sess.run(tf.global_variables_initializer())

    def train_network(self, batch_xs, batch_ys, learning_rate):
        """Train the network using a given batch

        Args:
            batch_xs: the input values
            batch_ys: the target values
            learning_rate: the learning rate for the optimiser

        Return:
            The current loss and summary
        """
        if learning_rate:
            self.learning_rate = learning_rate
        feed_dict = {
            self.net.x: batch_xs,
            self.lr: self.learning_rate,
            self.net.is_training: True
        }
        if batch_ys is not None:
            feed_dict.update({self.net.y_true: batch_ys})

        _, current_loss, train_loss_summary = self.sess.run(
            [self.train_step, self.loss_op, self.train_loss_summary],
            feed_dict=feed_dict)

        return current_loss, train_loss_summary

    def test_network(self, loader, epoch):
        """Test the network using the whole data

        Args:
            loader: the batch loader to use
            epoch: the current epoch, None if no writing is necessary

        Return:
            The test loss and accuracy
        """
        total_accuracy = 0
        total_loss = 0
        nb_batches = loader.num_batches
        loader.reset_batch_pointer()
        for i in range(nb_batches):
            batch_xs, batch_ys = loader.next_batch()
            feed_dict = {
                self.net.x: batch_xs,
                self.net.is_training: False
            }
            if batch_ys is not None:
                feed_dict.update({self.net.y_true: batch_ys})
            batch_loss, batch_accuracy, test_loss_summary, acc_summary = self.test_sess.run(
                [self.loss_op, self.net.accuracy, self.test_loss_summary, self.acc_summary], feed_dict=feed_dict)
            total_accuracy += batch_accuracy
            total_loss += batch_loss

            if epoch:
                self.writer.add_summary(test_loss_summary, epoch * nb_batches + i)
                self.writer.add_summary(acc_summary, epoch * nb_batches + i)

        return total_loss / nb_batches, total_accuracy / nb_batches

    def predict_sequence(self, batch_xs, batch_ys=None):
        """Predict a sequence from a batch of data, this uses the true values
        as input to the decoder.

        Args:
            batch_xs: the input values
            batch_ys: the target values

        Return:
            The predicted sequence
        """
        feed_dict = {self.net.x: batch_xs}
        if batch_ys is not None:
            feed_dict.update({self.net.y_true: batch_ys})
        return self.test_sess.run(self.net.predicted_sequence, feed_dict=feed_dict)

    def sample_sequence(self, batch_xs):
        """Sample a sequence from a batch of data.

        Args:
            batch_xs: the input values

        Return:
            The sampled sequence
        """
        feed_dict = {self.net.x: batch_xs}
        return self.test_sess.run(self.net.sampled_sequence, feed_dict=feed_dict)

    def kl_loss(self):
        """The KL divergence loss either with the standard normal or with learnable
        mean and variance of marginal.

        Return:
            The kl loss value
        """
        if self.net.update_marginal:
            kl_loss = kl_divergence(
                self.net.mu, self.net.sigma, self.net.mu0, self.net.sigma0)
        else:
            kl_loss = kl_divergence_with_std(self.net.mu, self.net.sigma)
        return kl_loss

    def loss(self):
        """Loss function: I(Y, Z) - beta * I(X, Z)

        Return:
            The loss value
        """
        reconstruction_loss = self.reconstruction_loss()

        if self.reduce_seq:
            reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=1)

        if self.beta:
            return tf.reduce_mean(reconstruction_loss + self.beta * self.kl_loss())
        return tf.reduce_mean(reconstruction_loss)

    @abstractmethod
    def reconstruction_loss(self):
        pass


class DiscreteLossLearner(Learner):
    """The loss learner for discrete data"""

    def __init__(self, network, beta, learning_rate, train_batch, run_name, binary=False,
                 reduce_seq=False):
        super().__init__(network, learning_rate, train_batch, beta, reduce_seq, run_name)

    def reconstruction_loss(self):
        """The cross entropy loss / log likelihood of categorical distribution"""

        if self.binary:
            return tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.net.y_true, logits=self.net.output)
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=self.net.y_true, logits=self.net.output)


class ContinuousLossLearner(Learner):
    """The loss learner for continuous data"""
    def __init__(self, network, beta, learning_rate, train_batch, run_name, reduce_seq=False):
        super().__init__(network, learning_rate, train_batch, beta, reduce_seq, run_name)

    def reconstruction_loss(self):
        """The log likelihood of normal distribution"""
        mu = self.net.output
        sigma = self.net.output_sigma
        return 0.5 * (
            tf.square(self.net.y_true - mu) / tf.square(sigma) + tf.log(2 * math.pi * tf.square(sigma)))
