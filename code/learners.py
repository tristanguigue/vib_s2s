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
    def __init__(self, network, learning_rate, train_batch, run_name):
        self.lr = tf.placeholder(tf.float32)
        self.net = network
        self.learning_rate = learning_rate
        self.train_batch = train_batch
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

    @abstractmethod
    def loss(self):
        pass

    def train_network(self, batch_xs, batch_ys, learning_rate):
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
        feed_dict = {self.net.x: batch_xs}
        if batch_ys is not None:
            feed_dict.update({self.net.y_true: batch_ys})
        return self.test_sess.run(self.net.predicted_sequence, feed_dict=feed_dict)

    def sample_sequence(self, batch_xs, batch_ys=None):
        feed_dict = {self.net.x: batch_xs}
        if batch_ys is not None:
            feed_dict.update({self.net.y_true: batch_ys})
        return self.test_sess.run(self.net.sampled_sequence, feed_dict=feed_dict)


class SupervisedLossLearner(Learner):
    def __init__(self, network, beta, learning_rate, train_batch, run_name, binary=False,
                 continuous=False, reduce_seq=False):
        self.beta = beta
        self.binary = binary
        self.continuous = continuous
        self.reduce_seq = reduce_seq
        super().__init__(network, learning_rate, train_batch, run_name)

    def loss(self):
        if self.binary:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.net.y_true, logits=self.net.output)
        elif self.continuous:
            mu = self.net.output
            sigma = self.net.output_sigma
            cross_entropy = 0.5 * (
                tf.square(self.net.y_true - mu) / tf.square(sigma) + tf.log(2 * math.pi * tf.square(sigma)))
        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.net.y_true, logits=self.net.output)
        if self.reduce_seq:
            cross_entropy = tf.reduce_mean(cross_entropy, axis=1)

        if self.net.update_prior:
            kl_loss = kl_divergence(
                self.net.mu, self.net.sigma, self.net.mu0, self.net.sigma0)
        else:
            kl_loss = kl_divergence_with_std(self.net.mu, self.net.sigma)

        if self.beta:
            return tf.reduce_mean(cross_entropy + self.beta * kl_loss)
        return tf.reduce_mean(cross_entropy)


class SequenceLossLearner(Learner):
    def __init__(self, network, beta, learning_rate, train_batch, run_name, binary=True):
        self.beta = beta
        self.binary = binary
        super().__init__(network, learning_rate, train_batch, run_name)

    def loss(self):
        if self.binary:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.net.y_true, logits=self.net.output)
        else:
            cross_entropy = tf.square(tf.norm(self.net.y_true - self.net.output, axis=1))

        if self.beta:
            mu = tf.reshape(self.net.mu, [-1, self.net.seq_size, self.net.bottleneck_size])
            sigma = tf.reshape(self.net.sigma, [-1, self.net.seq_size, self.net.bottleneck_size])

            if self.net.update_prior:
                kl = kl_divergence(mu, sigma, self.net.mu0, self.net.sigma0, sequence=True)
            else:
                kl = kl_divergence_with_std(mu, sigma, sequence=True)

            # To check
            kl = tf.reduce_sum(kl[:, :-1], axis=1)
            return tf.reduce_mean(tf.squeeze(cross_entropy) + self.beta * kl[:, :-1])
        return tf.reduce_mean(cross_entropy)

